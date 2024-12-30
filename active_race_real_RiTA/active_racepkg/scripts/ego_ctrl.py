#!/usr/bin/env python3
'''
MIT License

Copyright (c) 2024 High-Assurance Mobility and Control (HMC) Laboratory at Ulsan National Institute of Scienece and Technology (UNIST), Republic of Korea

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
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from active_racepkg.simulation.dynamics_simulator import DynamicsSimulator
from active_racepkg.dynamics.models.dynamics_models import CasadiDynamicBicycleFull
from active_racepkg.controllers.PID import PIDLaneFollower
from active_racepkg.controllers.utils.controllerTypes import PIDParams
from active_racepkg.controllers.MPCC_H2H_approx import MPCC_H2H_approx
from active_racepkg.common.pytypes import VehicleState, ParametricPose, BodyLinearVelocity, VehicleActuation
from active_racepkg.common.utils.file_utils import *
from active_racepkg.h2h_configs import *
from active_racepkg.path_generator import PathGenerator
from active_racepkg.utils import prediction_to_rosmsg, rosmsg_to_prediction, odom_to_pose, pose_to_vehicleState, odom_to_vehicleState, state_prediction_to_marker
from active_racemsgs.msg import VehiclePredictionROS
from collections import deque

class EgoCtrl(Node):
    def __init__(self):
        super().__init__('ego_controller_node')
        
        # Node Parameters
        self.n_nodes = self.declare_parameter('n_nodes', 12).value
        self.t_horizon = self.declare_parameter('t_horizon', 1.2).value
        self.dt = self.t_horizon / self.n_nodes

        self.use_predictions_from_module = True
        self.isitvelonce = False
        self.vel_cmd = 0.0

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
        
        # Publishers
        self.ackman_pub = self.create_publisher(AckermannDriveStamped, '/orinnx/ackermann_cmd_main', 2)
        self.ego_pred_pub = self.create_publisher(VehiclePredictionROS, "/ego_pred", 2)
        self.ego_pred_marker_pub = self.create_publisher(MarkerArray, "/ego_pred_marker", 2)

        # Subscribers
        self.tv_pred = None
        self.tv_pred_sub = self.create_subscription(VehiclePredictionROS, "/tar_pred", self.tar_pred_callback, 2)

        self.ego_odom_sub = self.create_subscription(Odometry, "/orinnx/tracked_odom", self.ego_odom_callback, 2)
        self.ego_pose_sub = self.create_subscription(PoseStamped, "/orinnx/tracked_pose", self.ego_pose_callback, 2)

        self.target_odom_sub = self.create_subscription(Odometry, "/nuc/tracked_odom", self.target_odom_callback, 2)
        self.target_pose_sub = self.create_subscription(PoseStamped, "/nuc/tracked_pose", self.target_pose_callback, 2)

        # Vehicle Model and Controller Initialization
        self.vehicle_model = CasadiDynamicBicycleFull(0.0, ego_dynamics_config, track=self.track_info.track)

        self.gp_mpcc_ego_controller = MPCC_H2H_approx(self.vehicle_model, self.track_info.track, control_params=gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name="test_track")
        self.gp_mpcc_ego_controller.initialize()

        # self.mpcc_ego_controller = MPCC_H2H_approx(self.vehicle_model, self.track_info.track, control_params=mpcc_ego_params, name="mpcc_h2h_ego", track_name="test_track")
        # self.mpcc_ego_controller.initialize()

        self.warm_start()

        self.pp_cmd = AckermannDriveStamped()

        # Timer for cmd_callback
        self.cmd_hz = 20
        self.cmd_timer = self.create_timer(1.0 / self.cmd_hz, self.cmd_callback)

    def tar_pred_callback(self, tar_pred_msg):
        self.tv_pred = rosmsg_to_prediction(tar_pred_msg)

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

    def warm_start(self, offset=0.0, approx=True):
        ego_sim_state = self.cur_ego_state.copy()

        ego_sim_state.p.s -= offset

        # self.track_info.track.local_to_global_typed(ego_sim_state)

        # self.track_info.track.update_curvature(tar_sim_state)

        state_history_ego = deque([], N); input_history_ego = deque([], N)

        input_ego = VehicleActuation()

        pid_steer_params = PIDParams()
        pid_steer_params.dt = self.dt
        pid_steer_params.default_steer_params()
        pid_steer_params.Kp = 1

        pid_speed_params = PIDParams()
        pid_speed_params.dt = self.dt
        pid_speed_params.default_speed_params()
        
        pid_controller_1 = PIDLaneFollower(ego_sim_state.v.v_long, ego_sim_state.p.x_tran, self.dt, pid_steer_params, pid_speed_params)
        
        ego_dynamics_simulator = DynamicsSimulator(0.0, ego_dynamics_config, track=self.track_info.track)

        n_iter = self.n_nodes + 1
        t = 0.0

        while n_iter > 0:
            pid_controller_1.step(ego_sim_state)

            ego_dynamics_simulator.step(ego_sim_state)

            self.track_info.track.update_curvature(ego_sim_state)
            
            input_ego.t = t
            ego_sim_state.copy_control(input_ego)
            q, _ = ego_dynamics_simulator.model.state2qu(ego_sim_state)
            u = ego_dynamics_simulator.model.input2u(input_ego)
            if approx:
                q = np.append(q, ego_sim_state.p.s)
                q = np.append(q, ego_sim_state.p.s)
                u = np.append(u, ego_sim_state.v.v_long)
            state_history_ego.append(q)
            input_history_ego.append(u)

            # self.track_info.track.update_curvature(tar_sim_state)

            n_iter -= 1
            t += self.dt

        compose_history = lambda state_history, input_history: (np.array(state_history), np.array(input_history))
        ego_warm_start_history = compose_history(state_history_ego, input_history_ego)

        self.gp_mpcc_ego_controller.set_warm_start(*ego_warm_start_history)
        # self.mpcc_ego_controller.set_warm_start(*ego_warm_start_history)

        self.get_logger().info("Warm start done!")

    def once_vel_cmd(self):
        if self.isitvelonce == False:
            self.vel_cmd = 0.5
            self.isitvelonce == True
        else:
            pass

    def cmd_callback(self):
        if self.ego_odom_ready and self.tar_odom_ready:
            pose_to_vehicleState(self.track_info.track, self.cur_ego_state, self.cur_ego_pose)
            odom_to_vehicleState(self.track_info.track, self.cur_ego_state, self.cur_ego_odom)
            pose_to_vehicleState(self.track_info.track, self.cur_tar_state, self.cur_tar_pose)
            odom_to_vehicleState(self.track_info.track, self.cur_tar_state, self.cur_tar_odom)
        else:
            self.get_logger().warn("State not ready!")
            return
        
        info, b, exitflag = self.gp_mpcc_ego_controller.step(ego_state=self.cur_ego_state, tv_state=self.cur_tar_state, tv_pred=self.tv_pred if self.use_predictions_from_module else None)
        ego_state_pred = self.gp_mpcc_ego_controller.get_prediction().copy()

        ego_pred_msg = prediction_to_rosmsg(ego_state_pred)
        self.ego_pred_pub.publish(ego_pred_msg)

        if ego_state_pred is not None and ego_state_pred.x is not None:
            if len(ego_state_pred.x) > 0:
                ego_marker_color = [0.0, 1.0, 0.0]
                ego_state_pred_marker = state_prediction_to_marker(ego_state_pred, ego_marker_color)
                self.ego_pred_marker_pub.publish(ego_state_pred_marker)

        self.pp_cmd.header.stamp = self.cur_ego_pose.header.stamp
        self.once_vel_cmd()

        if not info["success"]:
            self.get_logger().warn(f"EGO infeasible - Exitflag: {exitflag}")
        else:
            pred_v_lon = self.gp_mpcc_ego_controller.x_pred[:,0]
            cmd_accel = self.gp_mpcc_ego_controller.x_pred[1,9]

            # consider the delay in vel command
            if cmd_accel < 0.0:
                self.vel_cmd = pred_v_lon[6]
            else:
                self.vel_cmd = pred_v_lon[4]

        # if self.vel_cmd <= 0.1:
        #     self.vel_cmd = 0.3

        self.pp_cmd.drive.speed = 1.6 * self.vel_cmd
        self.pp_cmd.drive.steering_angle = 0.3 * self.cur_ego_state.u.u_steer
        self.ackman_pub.publish(self.pp_cmd)

def main(args=None):
    rclpy.init(args=args)
    ego_ctrl = EgoCtrl()
    rclpy.spin(ego_ctrl)
    ego_ctrl.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
