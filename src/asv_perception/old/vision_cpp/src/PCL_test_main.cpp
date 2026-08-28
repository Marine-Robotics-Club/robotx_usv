//  Filename:  acoustics.cpp
//  Creation Date:  05/22/2023
//  Last Revision Date:  05/23/2022
//  Author [email]:  Xavier Vicent [xvicentnavarro@lssu.edu]
//  Revisor(s) [Revision Date]:
//  Organization/Institution:  Lake Superior State University LSSU TEAM AMORE
// 
//
// ...............................About acoustics.cpp......................................

//
//		Inputs [subscribers]:
//		Outputs [publishers]: states of all other executables

//...............................................................................................Included Libraries and Message Types.........................................................................................
#include "vision_cpp/PCL_test.h"

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  while(rclcpp::ok())
  {
  
  	rclcpp::spin(std::make_shared<VRX_AMORE::PCL_test>());
  	rclcpp::shutdown();
  
  }
  return 0;
}
