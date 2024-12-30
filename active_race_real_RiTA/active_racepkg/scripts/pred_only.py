#!/usr/bin/env python3
'''
MIT License

Copyright (c) 2024 High-Assurance Mobility and Control (HMC) Laboratory at Ulsan National Institute of Science and Technology (UNIST), Republic of Korea

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
import time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from active_racemsgs.msg import VehiclePredictionROS
from message_filters import ApproximateTimeSynchronizer, Subscriber
from active_racepkg.path_generator import PathGenerator
from active_racepkg.common.pytypes import VehicleState, ParametricPose, BodyLinearVelocity, VehicleActuation
from active_racepkg.utils import odom_to_pose, prediction_to_rosmsg, rosmsg_to_prediction, pose_to_vehicleState, odom_to_vehicleState, prediction_to_marker, fill_global_info
from active_racepkg.prediction.trajectory_predictor import ConstantVelocityPredictor, ConstantAngularVelocityPredictor, NLMPCPredictor, GPPredictor
from active_racepkg.common.utils.file_utils import *
from active_racepkg.h2h_configs import *

class Predictor(Node):
    def __init__(self):
        super().__init__('predictor_node')

        # Node Parameters
        self.n_nodes = self.declare_parameter('n_nodes', 12).value
        self.t_horizon = self.declare_parameter('t_horizon', 1.2).value
        self.dt = self.t_horizon / self.n_nodes
        
        self.use_predictions_from_module = True

        # Track Info Generation
        self.track_info = PathGenerator()
        while not self.track_info.track_ready:
            rclpy.sleep(0.01)
        
        # Ego and Target States Initialization
        self.cur_ego_odom = Odometry()
        self.cur_ego_pose = PoseStamped()
        self.cur_ego_state = VehicleState(
            t=0.0,
            p=ParametricPose(s=0.1, x_tran=0.2, e_psi=0.0),
            v=BodyLinearVelocity(v_long=0.5),
            u=VehicleActuation(t=0.0, u_a=0.0, u_steer=0.0)
        )

        self.cur_tar_odom = Odometry()
        self.cur_tar_pose = PoseStamped()
        self.cur_tar_state = VehicleState()

        self.ego_odom_ready = False
        self.tar_odom_ready = False

        # Prediction and controller callback
        self.tv_pred = None
        self.tv_cav_pred = None
        self.tv_nmpc_pred = None
        self.tv_gp_pred = None

        self.ego_list = []
        self.tar_list = []
        self.tar_pred_list = []

        self.ego_pred = None
        self.gt_tar_pred = None
        self.ego_prev_pose = None
        self.tar_prev_pose = None
        self.ego_local_vel = None
        self.tar_local_vel = None

        # Publishers
        self.tar_pred_pub = self.create_publisher(VehiclePredictionROS, '/tar_pred', 2)
        self.tv_pred_marker_pub = self.create_publisher(MarkerArray, '/tv_pred_marker', 2)
        
        # Subscribers
        self.ego_pred_sub = self.create_subscription(VehiclePredictionROS, '/ego_pred', self.ego_pred_callback, 2)
        self.gt_tar_pred_sub = self.create_subscription(VehiclePredictionROS, '/gt_tar_pred', self.gt_tar_pred_callback, 2)

        self.ego_odom_sub = self.create_subscription(Odometry, "/orinnx/tracked_odom", self.ego_odom_callback, 2)
        self.target_odom_sub = self.create_subscription(Odometry, "/nuc/tracked_odom", self.target_odom_callback, 2)

        # Synchronization of pose messages
        self.ego_pose_sub = Subscriber(self, PoseStamped, "/orinnx/tracked_pose")
        self.target_pose_sub = Subscriber(self, PoseStamped, "/nuc/tracked_pose")
        self.ats = ApproximateTimeSynchronizer([self.ego_pose_sub, self.target_pose_sub], queue_size=2, slop=0.05)
        self.sync_prev_time = Clock().now()
        self.ats.registerCallback(self.sync_callback)

        # Select predictor
        self.predictor = None
        self.predictor_type = 1

        # 1. CAV
        self.cav_predictor = ConstantAngularVelocityPredictor(N=N, track=self.track_info.track, cov=0.01)
        # 2. NLMPC
        self.nlmpc_predictor = NLMPCPredictor(N=N, track=self.track_info.track, cov=0.02, v_ref=5.0)
        self.nlmpc_predictor.set_warm_start()
        # 3. Baseline GP
        self.base_gp_predictor = GPPredictor(N=N, track=self.track_info.track, policy_name="aggressive_blocking", use_GPU=True, M=20, cov_factor=np.sqrt(2))

        # Timer for prediction_callback
        self.cmd_hz = 20
        self.create_timer(1.0 / self.cmd_hz, self.prediction_callback)
        
    def ego_odom_callback(self, msg):
        self.ego_odom_ready = True
        self.cur_ego_odom = msg
        # self.cur_ego_pose = odom_to_pose(msg)

    def ego_pose_callback(self, msg):
        self.cur_ego_pose = msg

    def target_odom_callback(self, msg):
        self.tar_odom_ready = True
        self.cur_tar_odom = msg
        # self.cur_tar_pose = odom_to_pose(msg)

    def target_pose_callback(self, msg):
        self.cur_tar_pose = msg

    def ego_pred_callback(self, msg):
        self.ego_pred = rosmsg_to_prediction(msg)

    def gt_tar_pred_callback(self, msg):
        self.gt_tar_pred = rosmsg_to_prediction(msg)

    def sync_callback(self, ego_pose_msg, target_pose_msg):
        sync_cur_time = Clock().now()
        diff_sync_time = sync_cur_time - self.sync_prev_time
        if abs(diff_sync_time.nanoseconds / 1e9) > 0.05:
            self.get_logger().warn("sync diff time " + str(diff_sync_time.nanoseconds / 1e9))
        self.sync_prev_time = sync_cur_time
        self.ego_pose_callback(ego_pose_msg)
        self.target_pose_callback(target_pose_msg)

    def prediction_callback(self):
        start_time = time.time()
        
        if self.ego_odom_ready is False or self.tar_odom_ready is False or self.ego_pred is None:
            return
        
        if self.ego_odom_ready and self.tar_odom_ready:
            pose_to_vehicleState(self.track_info.track, self.cur_ego_state, self.cur_ego_pose)
            odom_to_vehicleState(self.track_info.track, self.cur_ego_state, self.cur_ego_odom)
            pose_to_vehicleState(self.track_info.track, self.cur_tar_state, self.cur_tar_pose)
            odom_to_vehicleState(self.track_info.track, self.cur_tar_state, self.cur_tar_odom)
            # self.cur_tar_state.v.v_long = self.cur_tar_state.v.v_long*1.2
        else:
            self.get_logger().warn("State not ready!")
            return

        if self.cur_ego_state.t is not None and self.cur_tar_state.t is not None and self.ego_pred.x is not None:

            if self.predictor_type == 0:
                self.tv_pred = self.cav_predictor.get_prediction(ego_state=self.cur_ego_state, target_state=self.cur_tar_state, ego_prediction=self.ego_pred)
            elif self.predictor_type == 1:
                self.tv_pred = self.nlmpc_predictor.get_prediction(ego_state=self.cur_ego_state, target_state=self.cur_tar_state, ego_prediction=self.ego_pred)
            elif self.predictor_type == 2:
                self.tv_pred = self.base_gp_predictor.get_prediction(ego_state=self.cur_ego_state, target_state=self.cur_tar_state, ego_prediction=self.ego_pred)

            tar_pred_msg = None

            if self.tv_pred is not None:
                fill_global_info(self.track_info.track, self.tv_pred)
                tar_pred_msg = prediction_to_rosmsg(self.tv_pred)
                tv_pred_markerArray = prediction_to_marker(self.tv_pred)
                
            if tar_pred_msg is not None:
                self.tar_pred_pub.publish(tar_pred_msg)
                self.tv_pred_marker_pub.publish(tv_pred_markerArray)

        end_time = time.time()

        execution_time = end_time - start_time
        if execution_time > 0.1:
            self.get_logger().info(f"Prediction execution time: {execution_time} seconds")

def main(args=None):
    rclpy.init(args=args)
    predictor = Predictor()
    rclpy.spin(predictor)
    predictor.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
