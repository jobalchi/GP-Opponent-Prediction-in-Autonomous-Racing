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
from visualization_msgs.msg import MarkerArray

from active_racepkg.h2h_configs import *
from active_racepkg.common.utils.file_utils import *
from active_racepkg.path_generator import PathGenerator

class TrajPub(Node):
    def __init__(self):   
        super().__init__('traj_pub_node')

        self.declare_parameter('n_nodes', 10)
        self.declare_parameter('t_horizon', 1.0)
        self.declare_parameter('traj_hz', 1)

        self.n_nodes = self.get_parameter('n_nodes').get_parameter_value().integer_value
        self.t_horizon = self.get_parameter('t_horizon').get_parameter_value().double_value
        self.traj_hz = self.get_parameter('traj_hz').get_parameter_value().integer_value

        self.center_pub = self.create_publisher(MarkerArray, '/center_line', 2)
        self.bound_in_pub = self.create_publisher(MarkerArray, '/track_bound_in', 2)
        self.bound_out_pub = self.create_publisher(MarkerArray, '/track_bound_out', 2)

        self.dt = self.t_horizon / float(self.n_nodes)

        # Generate Racing track info 
        self.track_info = PathGenerator()
        while not self.track_info.track_ready:
            rclpy.sleep(0.01)

        self.traj_pub_timer = self.create_timer(1.0 / self.traj_hz, self.traj_timer_callback)

    def traj_timer_callback(self):
        if self.track_info.track_ready:
            self.center_pub.publish(self.track_info.centerline)
            self.bound_in_pub.publish(self.track_info.track_bound_in)
            self.bound_out_pub.publish(self.track_info.track_bound_out)

def main(args=None):
    rclpy.init(args=args)
    node = TrajPub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
