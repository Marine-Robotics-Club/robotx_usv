//  Filename:  mission_control.cpp
//  Creation Date:  05/22/2023
//  Last Revision Date:  05/23/2022
//  Author [email]:  Xavier Vicent [xvicentnavarro@lssu.edu]
//  Revisor(s) [Revision Date]:
//  Organization/Institution:  Lake Superior State University LSSU TEAM AMORE
// 
//  Notes from author: PLEASE MAKE ANY PRINTOUTS START WITH "MISSION_CONTROL: " TO KNOW THAT IT IS FROM MISSION_CONTROL
//
// .................About mission_control.cpp......................................
//  This code acts as the autonomous state machine of the WAM-V USV.
//  It will get the parameters to control the state of the system.
//  This code will subscribe to goal poses given from the navigation_array.
//  Dependent on the current task state and system state, mission_control
//  will publish whether or not the low level controllers should be on.
//
//		Inputs [subscribers]:
//		Outputs [publishers]: states of all other executables

//all caps needs to be the same name as your desired .cpp file
#ifndef PCL_TEST_H
#define PCL_TEST_H

//...............................................................................................Included Libraries and Message Types.........................................................................................
//these are defaults always include these 
#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "time.h"
#include <sstream>
#include <iostream>
#include "math.h"
#include "stdio.h"
// define all needed libraries for your desired .cpp file 
#include "std_msgs/msg/bool.hpp" 
#include "std_msgs/msg/int64.hpp"  
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl_conversions.h"
#include "transforms.hpp"
#include "pcl_node.hpp"
//#include "pcl_ros/filters/voxel_grid.hpp"


//not sure what this does
using std::placeholders::_1;

//define alias's for cout and endl
using std::cout;
using std::endl;
using namespace std;

//unsure what namespace does yet .....
namespace VRX_AMORE {
//declaring the class name 
class PCL_test : public rclcpp::Node
{
 public:
 
 //declares the .cpp file XXXXX()
  PCL_test();
     
//Subscriber Init list all subscribers for desired .cpp files input to the Subscriber Function
	rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub;

 private:
 	
 	//Subscriber Functions ... the function that accesses and manipulates the data
	void cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

	//ALL Functions that are C++ not ros Subscriber Functions
	void loop();

	//rclcpp class
	rclcpp::Time start_time, current_time, last_time;  // creates time variables
	rclcpp::TimerBase::SharedPtr timer;
	
	//Define Publishers
	rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_test_pub;

	// C++ Global Variables
	int loop_count = 0;  
	
	//ROS2 variable declaration
	sensor_msgs::msg::PointCloud2 output;
	sensor_msgs::msg::PointCloud2 cloud_out;
};
}


#endif
