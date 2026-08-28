//  Filename:  acoustics.cpp
//  Creation Date:  05/22/2023
//  Last Revision Date:  05/23/2022
//  Author [email]:  Xavier Vicent [xvicentnavarro@lssu.edu]
//  Revisor(s) [Revision Date]:
//  Organization/Institution:  Lake Superior State University LSSU TEAM AMORE
// 
//  Notes from author: PLEASE MAKE ANY PRINTOUTS START WITH "MISSION_CONTROL: " TO KNOW THAT IT IS FROM MISSION_CONTROL
//
// ...............................About propulsion_system.cpp......................................

//		Inputs [subscribers]:
//		Outputs [publishers]: states of all other executables

//...............................................................................................Included Libraries and Message Types.........................................................................................

#include "vision_cpp/PCL_test.h"


	
VRX_AMORE::PCL_test::PCL_test ()
: Node("PCL_test")
{
	//AMORE Subscriptions
	cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(		
	"/wamv/sensors/lidars/lidar_wamv_sensor/points", 1, std::bind(&PCL_test::cloud_cb, this, _1));

	pcl_test_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("vision/pc2/test", 10);
	
	//essentially is equal to setting the loop rate 500ms 
	timer = this->create_wall_timer(500ms, std::bind(&PCL_test::loop, this));
	     
}

void VRX_AMORE::PCL_test::cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg){

	pcl::PCLPointCloud2::Ptr voxel (new pcl::PCLPointCloud2 ());
	RCLCPP_INFO(this->get_logger(), "I received the message");
	pcl_conversions::toPCL(*msg,*voxel);    


	pcl_conversions::fromPCL(*voxel,cloud_out);  

	//unsigned int num_points_out = cloud_out.width;

	cloud_out.header.frame_id = msg->header.frame_id;
	cloud_out.header.stamp = msg->header.stamp;

	
}



//Main Loop
void VRX_AMORE::PCL_test::loop()
{
	pcl_test_pub->publish(cloud_out);
}

