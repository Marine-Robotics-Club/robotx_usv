#include "pid_hs/pid_hs.h"

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  
  while(rclcpp::ok())
  {
  
  	rclcpp::spin(std::make_shared<FAU::PID_HS>());
  	rclcpp::shutdown();
  
  }
  return 0;
  
}
