#ifndef RB26_H
#define RB26_H

#include <Arduino.h>

// =========================================================
// PINS / CONSTANTS
// =========================================================
#define PORT_PW       23
#define STBD_PW       3
#define T500_EN       38
#define ESTOP_SENSE   A15
#define FAN_EN        37
#define BATT_SENS_EN  33
#define LT_ORANGE     30
#define LT_GREEN      31
#define LT_RED        27
#define TEMP          A12
#define BATT_V        A3

#define CH1  1   // inverted PWM
#define CH2  0   // inverted PWM
#define CH3  6   // inverted PWM (unused)
#define CH4  7   // inverted PWM
#define CH5  4   // inverted PWM
#define SBUS_CH6 5

#define Jetson Serial

#define PWM_MIN      1100
#define PWM_MAX      1900
#define PWM_NEUTRAL  1500

static const uint32_t TELEMETRY_PERIOD_MS = 100;

// =========================================================
// GLOBAL STATE (extern)
// =========================================================
extern volatile float accelX, accelY, accelZ;
extern volatile float gyroX, gyroY, gyroZ;
extern volatile float quatW, quatX, quatY, quatZ;

extern volatile float temperature;
extern volatile float systemBattery;
extern volatile float e_stop;

extern volatile bool isAutoFlag;
extern volatile bool isKilledFlag;
extern volatile bool wasD_Estop;
extern volatile bool wasKilledFlag;
extern volatile bool autoModeTransitionFlag;
extern volatile bool killedRecoveredFlag;

extern volatile uint16_t ch1, ch2, ch4, ch5;
extern volatile uint16_t port_pwm, stbd_pwm;

extern volatile int16_t ROS_PortThrust;
extern volatile int16_t ROS_StbdThrust;

extern volatile char pump_cmd;

extern volatile bool wasAutoFlag;
extern volatile bool wasKilledFlag;



// =========================================================
// FUNCTION PROTOTYPES
// =========================================================
void printGPS_All();
void printIMU();

void RC_Reads();
void AnalogReads();
void Mode_Check();
void lightTower();
void motor_start_auto();
void sendMotorCmds();

void service_jetson_rx();
void apply_command(int port_cmd, int stbd_cmd,
                   int estop, int auto_send, int kill_send,
                   char pump_char);

void send_jetson_telemetry();
void RB26_hw_init();
void RB26_gps_init(HardwareSerial &gpsSerial, uint32_t baud);
void RB26_pump_init(uint32_t baud);
void RB26_servos_init();
bool RB26_imu_init();
void RB26_rc_init();

void RB26_update_gps();
void RB26_update_imu();



#endif
