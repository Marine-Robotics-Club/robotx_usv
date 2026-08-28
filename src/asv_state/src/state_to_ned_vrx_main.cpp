#include "asv_state/state_to_ned_vrx.h"


int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  while(rclcpp::ok())
  {
  
  rclcpp::spin(std::make_shared<state_to_ned>());
  rclcpp::shutdown();
  
  }
}

