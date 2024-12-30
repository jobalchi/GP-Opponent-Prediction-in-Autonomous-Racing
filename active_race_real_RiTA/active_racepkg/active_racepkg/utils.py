#!/usr/bin/env python3
"""   
 Software License Agreement (BSD License)
 Copyright (c) 2022 Ulsan National Institute of Science and Technology (UNIST)
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************   
"""
import array
import rclpy
from rclpy.time import Time
from rclpy.clock import Clock
from rclpy.duration import Duration
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from active_racepkg.common.tracks.radius_arclength_track import RadiusArclengthTrack
from active_racepkg.common.pytypes import VehicleState, VehiclePrediction
from active_racemsgs.msg import VehiclePredictionROS
from active_racepkg.common.utils.file_utils import *
from geometry_msgs.msg import PoseStamped
from tf_transformations import euler_from_quaternion

def odom_to_pose(odom : Odometry):
    pose = PoseStamped()
    pose.header = odom.header
    pose.pose.position = odom.pose.pose.position
    pose.pose.orientation = odom.pose.pose.orientation
    return pose

def pose_to_vehicleState(track : RadiusArclengthTrack, state : VehicleState, pose : PoseStamped, line=None):
    state.x.x = pose.pose.position.x
    state.x.y = pose.pose.position.y

    orientation_q = pose.pose.orientation
    quat = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    cur_roll, cur_pitch, cur_yaw = euler_from_quaternion(quat)
    state.e.psi = cur_yaw

    xy_coord = (state.x.x, state.x.y, state.e.psi)
    cl_coord = track.global_to_local(xy_coord, line=line)

    if cl_coord is None:
        print('cl_coord is none')
        return
    
    # state.t = pose.header.stamp.to_sec()
    state.t = Time(seconds=pose.header.stamp.sec, nanoseconds=pose.header.stamp.nanosec).nanoseconds / 1e9
    state.p.s = cl_coord[0]
    ##################
    ## WHen track is doubled... wrap with half track length
    if state.p.s > track.track_length:
        state.p.s -= track.track_length
    ###################
    state.p.x_tran = cl_coord[1]
    state.p.e_psi = cl_coord[2]
    track.update_curvature(state)
    
def odom_to_vehicleState(track : RadiusArclengthTrack, state : VehicleState, odom : Odometry):
    local_vel = get_local_vel(odom, is_odom_local_frame=True)
    if local_vel is None:
        return
    
    state.v.v_long = local_vel[0]
    state.v.v_tran = local_vel[1]
    state.w.w_psi = odom.twist.twist.angular.z

def quaternion_to_rotation_matrix(quat):
        """
        Converts a quaternion to a 3x3 rotation matrix.
        """
        x, y, z, w = quat
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])

def get_local_vel(odom, is_odom_local_frame):
    local_vel = np.array([0.0, 0.0, 0.0])
    if is_odom_local_frame is False:
        # Extract the original linear velocity in odom frame
        odom_v = np.array([
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z
        ])
        # Extract quaternion and compute rotation matrix
        quat = [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]
        rotation_matrix = quaternion_to_rotation_matrix(quat)
        # Transform odom velocity
        local_vel = rotation_matrix @ odom_v  # Matrix multiplication
    else:
        local_vel[0] = odom.twist.twist.linear.x
        local_vel[1] = odom.twist.twist.linear.y
        local_vel[2] = odom.twist.twist.linear.z
    return local_vel

def fill_global_info(track, pred):
    if pred.s is not None and len(pred.s) > 0:
        pred.x = np.zeros(len(pred.s))
        pred.y = np.zeros(len(pred.s))
        pred.psi = np.zeros(len(pred.s))
        for i in range(len(pred.s)):
            cl_coord = [pred.s[i], pred.x_tran[i], pred.e_psi[i]]
            gl_coord = track.local_to_global(cl_coord)
            pred.x[i] = gl_coord[0]
            pred.y[i] = gl_coord[1]
            pred.psi[i] = gl_coord[2]

def prediction_to_marker(predictions):
    pred_path_marker_array = MarkerArray()
    if predictions is None or predictions.x is None:
        return pred_path_marker_array
    if len(predictions.x) <= 0:
        return pred_path_marker_array
    for i in range(len(predictions.x)):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.header.stamp = Clock().now().to_msg()
        marker_ref.ns = "pred"
        marker_ref.id = i
        marker_ref.type = Marker.SPHERE
        marker_ref.action = Marker.ADD
        marker_ref.pose.position.x = predictions.x[i]
        marker_ref.pose.position.y = predictions.y[i]
        marker_ref.pose.position.z = 0.0

        marker_ref.lifetime = Duration(seconds=0.2).to_msg()
        # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
        scale = 1
        if predictions.xy_cov is not None and len(predictions.xy_cov) != 0:
            x_cov = max(predictions.xy_cov[i][0,0],0.01)
            y_cov = max(predictions.xy_cov[i][1,1],0.01)
        else:
            x_cov = 0.01
            y_cov = 0.01
        marker_ref.scale.x = 2*np.sqrt(x_cov)*scale
        marker_ref.scale.y = 2*np.sqrt(y_cov)*scale
        marker_ref.scale.z = 0.1
        # high uncertainty will get red
        uncertainty_level = y_cov + x_cov
        # print(uncertainty_level)
        old_range_min, old_range_max = 0.0, 0.1+0.03*i
        new_range_min, new_range_max = 0, 1
        new_uncertainty = (uncertainty_level - old_range_min) * (new_range_max - new_range_min) / (old_range_max - old_range_min)
        new_uncertainty = max(min(new_uncertainty, new_range_max), new_range_min)  # Clamping to the new range
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (new_uncertainty, 1-new_uncertainty, 0.0)
        marker_ref.color.a = 0.2
        pred_path_marker_array.markers.append(marker_ref)
        
    return pred_path_marker_array

def state_prediction_to_marker(predictions, color):
    pred_path_marker_array = MarkerArray()
    if predictions is None or predictions.x is None:
        return pred_path_marker_array
    if len(predictions.x) <= 0:
        return pred_path_marker_array
    for i in range(len(predictions.x)):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"
        marker_ref.header.stamp = Clock().now().to_msg()
        marker_ref.ns = "pred"
        marker_ref.id = i
        marker_ref.type = Marker.SPHERE
        marker_ref.action = Marker.ADD
        marker_ref.pose.position.x = predictions.x[i]
        marker_ref.pose.position.y = predictions.y[i]
        marker_ref.pose.position.z = 0.0
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (color[0], color[1], color[2])
        marker_ref.color.a = 0.2
        marker_ref.lifetime = Duration(seconds=0.2).to_msg()
        # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
        scale = 1
        if predictions.xy_cov is not None:
            x_cov = max(predictions.xy_cov[i][0,0],0.00001)
            y_cov = max(predictions.xy_cov[i][1,1],0.00001)
        else:
            x_cov = 0.01
            y_cov = 0.01
        marker_ref.scale.x = 2 * np.sqrt(x_cov) * scale
        marker_ref.scale.y = 2 * np.sqrt(y_cov) * scale
        marker_ref.scale.z = 0.1
        pred_path_marker_array.markers.append(marker_ref)
    return pred_path_marker_array

def prediction_to_rosmsg(vehicle_prediction_obj : VehiclePrediction):
    ros_msg = VehiclePredictionROS()
    ros_msg.header.stamp = Clock().now().to_msg()
    ros_msg.header.frame_id = "map"
    # Assign values from the VehiclePrediction object to the ROS message
    # ros_msg.t = ros_msg.header.stamp.to_sec()
    ros_msg.t = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec / 1e9
    
    if vehicle_prediction_obj.x is not None:
        ros_msg.x = array.array('f', vehicle_prediction_obj.x)
    if vehicle_prediction_obj.y is not None:
        ros_msg.y = array.array('f', vehicle_prediction_obj.y)
    if vehicle_prediction_obj.v_x is not None:
        ros_msg.v_x = array.array('f', vehicle_prediction_obj.v_x)
    if vehicle_prediction_obj.v_y is not None:
        ros_msg.v_y = array.array('f', vehicle_prediction_obj.v_y)
    if vehicle_prediction_obj.a_x is not None:
        ros_msg.a_x = array.array('f', vehicle_prediction_obj.a_x)
    if vehicle_prediction_obj.a_y is not None:
        ros_msg.a_y = array.array('f', vehicle_prediction_obj.a_y)
    if vehicle_prediction_obj.psi is not None:
        ros_msg.psi = array.array('f', vehicle_prediction_obj.psi)
    if vehicle_prediction_obj.psidot is not None:
        ros_msg.psidot = array.array('f', vehicle_prediction_obj.psidot)
    if vehicle_prediction_obj.v_long is not None:
        ros_msg.v_long = array.array('f', vehicle_prediction_obj.v_long)
    if vehicle_prediction_obj.v_tran is not None:
        ros_msg.v_tran = array.array('f', vehicle_prediction_obj.v_tran)
    if vehicle_prediction_obj.a_long is not None:
        ros_msg.a_long = array.array('f', vehicle_prediction_obj.a_long)
    if vehicle_prediction_obj.a_tran is not None:
        ros_msg.a_tran = array.array('f', vehicle_prediction_obj.a_tran)
    if vehicle_prediction_obj.e_psi is not None:
        ros_msg.e_psi = array.array('f', vehicle_prediction_obj.e_psi)
    if vehicle_prediction_obj.s is not None:
        ros_msg.s = array.array('f', vehicle_prediction_obj.s)
    if vehicle_prediction_obj.x_tran is not None:
        ros_msg.x_tran = array.array('f', vehicle_prediction_obj.x_tran)
    if vehicle_prediction_obj.u_a is not None:
        ros_msg.u_a = array.array('f', vehicle_prediction_obj.u_a)
    if vehicle_prediction_obj.u_steer is not None:
        ros_msg.u_steer = array.array('f', vehicle_prediction_obj.u_steer)
    if vehicle_prediction_obj.lap_num is not None:
        ros_msg.lap_num = int(vehicle_prediction_obj.lap_num)
    if vehicle_prediction_obj.sey_cov is not None:
        ros_msg.sey_cov = vehicle_prediction_obj.sey_cov.tolist()
    if vehicle_prediction_obj.xy_cov is not None:
        xy_cov_1d = np.array(vehicle_prediction_obj.xy_cov).reshape(-1)
        ros_msg.xy_cov = xy_cov_1d.tolist()
    return ros_msg

def rosmsg_to_prediction(ros_msg : VehiclePredictionROS):
    vehicle_prediction_obj = VehiclePrediction()

    # Assign values from the ROS message to the VehiclePrediction object
    vehicle_prediction_obj.t = ros_msg.t
    vehicle_prediction_obj.x = array.array('f', ros_msg.x)
    vehicle_prediction_obj.y = array.array('f', ros_msg.y)
    vehicle_prediction_obj.v_x = array.array('f', ros_msg.v_x)
    vehicle_prediction_obj.v_y = array.array('f', ros_msg.v_y)
    vehicle_prediction_obj.a_x = array.array('f', ros_msg.a_x)
    vehicle_prediction_obj.a_y = array.array('f', ros_msg.a_y)
    vehicle_prediction_obj.psi = array.array('f', ros_msg.psi)
    vehicle_prediction_obj.psidot = array.array('f', ros_msg.psidot)
    vehicle_prediction_obj.v_long = array.array('f', ros_msg.v_long)
    vehicle_prediction_obj.v_tran = array.array('f', ros_msg.v_tran)
    vehicle_prediction_obj.a_long = array.array('f', ros_msg.a_long)
    vehicle_prediction_obj.a_tran = array.array('f', ros_msg.a_tran)
    vehicle_prediction_obj.e_psi = array.array('f', ros_msg.e_psi)
    vehicle_prediction_obj.s = array.array('f', ros_msg.s)
    vehicle_prediction_obj.x_tran = array.array('f', ros_msg.x_tran)
    vehicle_prediction_obj.u_a = array.array('f', ros_msg.u_a)
    vehicle_prediction_obj.u_steer = array.array('f', ros_msg.u_steer)
    vehicle_prediction_obj.lap_num = ros_msg.lap_num
    vehicle_prediction_obj.sey_cov = np.array(ros_msg.sey_cov)
    vehicle_prediction_obj.xy_cov = np.array(ros_msg.xy_cov).reshape(-1,2,2)

    return vehicle_prediction_obj
        