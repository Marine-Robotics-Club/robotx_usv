// main.ino (or RB26_Main.ino)
// RoboBoat 2026 Teensy 4.1 motherboard main sketch
// Only setup() and loop() live here.

#include <Arduino.h>

#include <Servo.h>
#include <elapsedMillis.h>

#include "RB26.h"
#include "GarminNMEA.h"
#include "IMUReader.h"
#include "READ_RC.h"

// If your build complains about these being multiply-defined,
// make sure they are ONLY defined in RB26.cpp (not in the .ino).
extern GarminNMEA garmin;
extern IMUReader imuReader;


void setup() {
  Jetson.begin(9600);

  RB26_hw_init();
  RB26_gps_init(Serial6, 38400);
  RB26_pump_init(9600);
  RB26_servos_init();

  if (!RB26_imu_init()) {
    Jetson.println("IMU not detected!");
    while (1) {}
  }

  RB26_rc_init();

  delay(2000);

  wasKilledFlag = isKilledFlag;
  wasAutoFlag   = isAutoFlag;
}

void loop() {
  RB26_update_gps();
  service_jetson_rx();
  RB26_update_imu();

  RC_Reads();
  AnalogReads();
  Mode_Check();
  lightTower();
  motor_start_auto();
  sendMotorCmds();

  Serial3.print(pump_cmd);
  send_jetson_telemetry();

  wasKilledFlag = isKilledFlag;
  wasAutoFlag   = isAutoFlag;
}

