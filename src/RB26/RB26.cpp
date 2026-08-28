// RB26.cpp
// RoboBoat 2026 Teensy 4.1 motherboard logic
// Put ALL function bodies + global definitions here.
// Keep setup()/loop() in your .ino.

#include "RB26.h"

#include <Arduino.h>
#include <Servo.h>
#include <elapsedMillis.h>

#include "GarminNMEA.h"
#include "IMUReader.h"
#include "READ_RC.h"

// =========================================================
//                 MODULE-LOCAL OBJECTS
// =========================================================
GarminNMEA garmin;

static Servo portThrust;
static Servo stbdThrust;

static elapsedMillis printTimer;
static elapsedMillis readAnalogs;
static elapsedMillis enableBattery;

IMUReader imuReader;

// =========================================================
//                 GLOBAL DEFINITIONS (ONCE)
//  NOTE: These MUST be declared as `extern` in RB26.h
// =========================================================

// ---- IMU ----
volatile float accelX = 0, accelY = 0, accelZ = 0;
volatile float gyroX  = 0, gyroY  = 0, gyroZ  = 0;
volatile float magX   = 0, magY   = 0, magZ   = 0;
volatile float eulerX = 0, eulerY = 0, eulerZ = 0;
volatile float quatW  = 1, quatX  = 0, quatY  = 0, quatZ = 0;
volatile float linX   = 0, linY   = 0, linZ   = 0;
volatile float gravX  = 0, gravY  = 0, gravZ  = 0;

// ---- RC RAW ----
volatile uint16_t RC_CH1 = 1500;
volatile uint16_t RC_CH2 = 1500;
volatile uint16_t RC_CH3 = 1500;
volatile uint16_t RC_CH4 = 1500;
volatile uint16_t RC_CH5 = 1500;
volatile uint16_t RC_CH6 = 1500;

// ---- System / status ----
volatile float temperature   = 0.0f;
volatile float systemBattery = 0.0f;
volatile float e_stop        = 0.0f;

volatile bool isAutoFlag   = false;
volatile bool isKilledFlag = true;   // start safe
volatile bool wasD_Estop   = false;

volatile bool killedRecoveredFlag     = false;
volatile bool timerStarted            = false;
volatile bool autoModeTransitionFlag  = false;

unsigned long servoStartTime = 0;

volatile char pump_cmd = 'a';   // default OFF

// ---- Jetson (ROS) commanded thrust ----
volatile int16_t ROS_StbdThrust = PWM_NEUTRAL;
volatile int16_t ROS_PortThrust = PWM_NEUTRAL;

// ---- RC live channels used in logic ----
volatile uint16_t ch1 = 1500;
volatile uint16_t ch2 = 1500;
volatile uint16_t ch4 = 1500;
volatile uint16_t ch5 = 1500;

// ---- Motor outputs (post mapping) ----
volatile uint16_t port_pwm = PWM_NEUTRAL;
volatile uint16_t stbd_pwm = PWM_NEUTRAL;

// ---- Jetson serial RX state ----
char rx_buf[128];
int  rx_len = 0;

volatile bool validChecksum     = false;
volatile uint32_t lastJetsonRxMs = 0;
unsigned long previousMillis = 0;

volatile bool wasAutoFlag   = false;
volatile bool wasKilledFlag = true;


// =========================================================
//                 OPTIONAL RAW NMEA CALLBACK
// =========================================================
static void onNmeaLine(const char* line) {
  // Optional: forward raw NMEA to USB serial
  // Jetson.println(line);
  (void)line;
}

// =========================================================
//                      GPS PRINT
// =========================================================
void printGPS_All()
{
  const auto& d = garmin.data();

  Jetson.println(F("===== GPS DATA ====="));

  // ---- RMC ----
  Jetson.print(F("RMC valid: "));
  Jetson.println(d.rmc_valid ? F("YES") : F("NO"));

  Jetson.print(F("Time (hhmmss): "));
  Jetson.println(d.time_hhmmss);

  Jetson.print(F("Date (ddmmyy): "));
  Jetson.println(d.date_ddmmyy);

  Jetson.print(F("Latitude (deg): "));
  Jetson.println(d.lat_deg, 8);

  Jetson.print(F("Longitude (deg): "));
  Jetson.println(d.lon_deg, 8);

  Jetson.print(F("Speed over ground (knots): "));
  Jetson.println(d.sog_knots, 3);

  Jetson.print(F("Course over ground (deg): "));
  Jetson.println(d.cog_deg, 2);

  // ---- GGA ----
  Jetson.print(F("Fix quality: "));
  Jetson.println(d.fix_quality);

  Jetson.print(F("Satellites used: "));
  Jetson.println(d.sats_used);

  Jetson.print(F("HDOP: "));
  Jetson.println(d.hdop, 2);

  Jetson.print(F("Altitude (m): "));
  Jetson.println(d.alt_m, 2);

  // ---- GSA ----
  Jetson.print(F("GSA valid: "));
  Jetson.println(d.gsa_valid ? F("YES") : F("NO"));

  Jetson.print(F("Fix type (1/2/3): "));
  Jetson.println(d.fix_type);

  Jetson.print(F("PDOP: "));
  Jetson.println(d.pdop, 2);

  Jetson.print(F("VDOP: "));
  Jetson.println(d.vdop, 2);

  // ---- GSV ----
  Jetson.print(F("Satellites in view: "));
  Jetson.println(d.sats_in_view);

  // ---- VTG ----
  Jetson.print(F("Speed (km/h): "));
  Jetson.println(d.speed_kmh, 3);

  // ---- PGRME ----
  Jetson.print(F("PGRME valid: "));
  Jetson.println(d.pgrme_valid ? F("YES") : F("NO"));

  Jetson.print(F("EPE horizontal (m): "));
  Jetson.println(d.epe_horz_m, 2);

  Jetson.print(F("EPE vertical (m): "));
  Jetson.println(d.epe_vert_m, 2);

  Jetson.print(F("EPE spherical (m): "));
  Jetson.println(d.epe_sphr_m, 2);

  Jetson.print(F("Last update (ms ago): "));
  Jetson.println(millis() - d.last_sentence_ms);

  Jetson.println(F("===================="));
}

// =========================================================
//                      IMU PRINT
// =========================================================
void printIMU() {
  Jetson.println("=== IMU DATA ===");

  Jetson.print("Euler (deg): ");
  Jetson.print(eulerX, 2); Jetson.print(", ");
  Jetson.print(eulerY, 2); Jetson.print(", ");
  Jetson.println(eulerZ, 2);

  Jetson.print("Accel (m/s^2): ");
  Jetson.print(accelX, 2); Jetson.print(", ");
  Jetson.print(accelY, 2); Jetson.print(", ");
  Jetson.println(accelZ, 2);

  Jetson.print("Gyro (rad/s): ");
  Jetson.print(gyroX, 2); Jetson.print(", ");
  Jetson.print(gyroY, 2); Jetson.print(", ");
  Jetson.println(gyroZ, 2);

  Jetson.print("Mag (uT): ");
  Jetson.print(magX, 2); Jetson.print(", ");
  Jetson.print(magY, 2); Jetson.print(", ");
  Jetson.println(magZ, 2);

  Jetson.print("Quat (w,x,y,z): ");
  Jetson.print(quatW, 3); Jetson.print(", ");
  Jetson.print(quatX, 3); Jetson.print(", ");
  Jetson.print(quatY, 3); Jetson.print(", ");
  Jetson.println(quatZ, 3);

  Jetson.print("Lin Acc (m/s^2): ");
  Jetson.print(linX, 2); Jetson.print(", ");
  Jetson.print(linY, 2); Jetson.print(", ");
  Jetson.println(linZ, 2);

  Jetson.print("Gravity (m/s^2): ");
  Jetson.print(gravX, 2); Jetson.print(", ");
  Jetson.print(gravY, 2); Jetson.print(", ");
  Jetson.println(gravZ, 2);

  Jetson.println();
}

// =========================================================
//                 PWM MAPPING
// =========================================================
static inline int map_pwm_rc_to_t500(int rc_us)
{
  rc_us = constrain(rc_us, 1000, 2000);

  if (rc_us < 1500) {
    // reverse: 1000..1500 -> 1100..1500
    return map(rc_us, 1000, 1500, 1100, 1500);
  } else {
    // forward: 1500..2000 -> 1500..1900
    return map(rc_us, 1500, 2000, 1500, 1900);
  }
}

// =========================================================
//                 LIGHT TOWER
// =========================================================
void lightTower()
{
  if (isKilledFlag)
  {
    digitalWrite(LT_RED, HIGH);
    digitalWrite(LT_ORANGE, LOW);
    digitalWrite(LT_GREEN, LOW);
  }
  else
  {
    if (isAutoFlag)
    {
      digitalWrite(LT_RED, LOW);
      digitalWrite(LT_GREEN, HIGH);
      digitalWrite(LT_ORANGE, LOW);
    }
    else
    {
      digitalWrite(LT_RED, LOW);
      digitalWrite(LT_GREEN, LOW);
      digitalWrite(LT_ORANGE, HIGH);
    }
  }
}

// =========================================================
//                 SEND MOTOR CMDS
// =========================================================
void sendMotorCmds()
{
  if (!isKilledFlag)
  {
    if (isAutoFlag && !autoModeTransitionFlag)
    {
      port_pwm = map_pwm_rc_to_t500(ROS_PortThrust);
      stbd_pwm = map_pwm_rc_to_t500(ROS_StbdThrust);
    }
    else
    {
      port_pwm = map_pwm_rc_to_t500(ch4);
      stbd_pwm = map_pwm_rc_to_t500(ch1);
    }

    portThrust.writeMicroseconds(port_pwm);
    stbdThrust.writeMicroseconds(stbd_pwm);
  }
}

// =========================================================
//                 MODE CHECK / SAFETY
// =========================================================
void Mode_Check()
{
  e_stop = (float)analogRead(ESTOP_SENSE);

  if ((ch2 > 1500) && (wasD_Estop))
  {
    digitalWrite(T500_EN, HIGH);
    wasD_Estop = false;
  }

  if ((e_stop < 2000.00) || (ch2 < 1100))
  {
    isKilledFlag = true;
    if (ch2 < 1100)
    {
      wasD_Estop = true;
      digitalWrite(T500_EN, LOW);
    }
  }
  else
  {
    isKilledFlag = false;
  }

  if (!isKilledFlag)
  {
    if (ch5 < 1100)
      isAutoFlag = true;
    else
      isAutoFlag = false;
  }
}

// =========================================================
//                 ANALOG READS
// =========================================================
void AnalogReads()
{
  if (enableBattery > 980)
  {
    digitalWrite(BATT_SENS_EN, HIGH);
    enableBattery = 0;
  }

  if (readAnalogs > 1000)
  {
    temperature = (((float)analogRead(TEMP) * 3.3f / 1024.0f) - 0.6f) / 0.1f;
    if (temperature > 40.0f) digitalWrite(FAN_EN, HIGH);
    else                     digitalWrite(FAN_EN, LOW);

    systemBattery = ((float)analogRead(BATT_V) * 3.3f / 1024.0f) * 8.96f / 4.0f;

    readAnalogs = 0;
    enableBattery = 0;
    digitalWrite(BATT_SENS_EN, LOW);
  }
}

// =========================================================
//                 RC READS
// =========================================================
void RC_Reads()
{
  ch1 = READ_RC(1);
  ch2 = READ_RC(2);
  ch4 = READ_RC(4);
  ch5 = READ_RC(5);
}

// =========================================================
//            MOTOR START AUTO / TRANSITION HOLD
// =========================================================
void motor_start_auto()
{
  // Detect transition from KILLED to ACTIVE
  if (wasKilledFlag && !isKilledFlag) {
    killedRecoveredFlag = true;
    servoStartTime = millis();
    timerStarted = true;
  }

  // Detect transition from MANUAL to AUTONOMOUS (your original condition)
  if ((wasKilledFlag && isAutoFlag)) {
    autoModeTransitionFlag = true;
    servoStartTime = millis();
    timerStarted = true;
  }

  // Hold neutral for 3 seconds after transitions
  if ((killedRecoveredFlag || autoModeTransitionFlag) && timerStarted) {

    portThrust.writeMicroseconds(1500);
    stbdThrust.writeMicroseconds(1500);

    if (millis() - servoStartTime >= 3000) {
      killedRecoveredFlag = false;
      autoModeTransitionFlag = false;
      timerStarted = false;

      ROS_PortThrust = 1500;
      ROS_StbdThrust = 1500;
    }
  }
}

// =========================================================
//                 JETSON FRAME HELPERS
// =========================================================
static inline bool isHexDigitChar(char c) {
  return (c >= '0' && c <= '9') ||
         (c >= 'a' && c <= 'f') ||
         (c >= 'A' && c <= 'F');
}

static uint8_t hex2_u8(const char* s2) {
  auto hexval = [](char c)->uint8_t {
    if (c >= '0' && c <= '9') return (uint8_t)(c - '0');
    if (c >= 'a' && c <= 'f') return (uint8_t)(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return (uint8_t)(c - 'A' + 10);
    return 0;
  };
  return (uint8_t)((hexval(s2[0]) << 4) | hexval(s2[1]));
}

static uint8_t checksum_cmd(int port_thrust,
                            int stbd_thrust,
                            int estop,
                            int auto_send,
                            int kill_send,
                            char pump_char)
{
  long sum = (long)port_thrust +
             (long)stbd_thrust +
             (long)estop +
             (long)auto_send +
             (long)kill_send +
             (long)((uint8_t)pump_char);

  if (sum < 0) sum = -sum;
  return (uint8_t)(sum % 256);
}

static bool parse_jetson_frame(const char* frame,
                               int &port_cmd, int &stbd_cmd,
                               int &estop, int &auto_send, int &kill_send,
                               char &pump_char,
                               uint8_t &rx_cc)
{
  const char* start = strchr(frame, '<');
  const char* end   = strchr(frame, '>');
  if (!start || !end || end <= start) return false;

  const char* star = strchr(start, '*');
  if (!star || star > end) return false;

  if ((end - star) < 3) return false; // needs "*hh"
  char h0 = star[1];
  char h1 = star[2];
  if (!isHexDigitChar(h0) || !isHexDigitChar(h1)) return false;

  char cc_str[3] = {h0, h1, 0};
  rx_cc = hex2_u8(cc_str);

  // Find comma before '*'
  const char* cs = star;
  while (cs > start && *cs != ',') cs--;
  if (*cs != ',') return false;

  // Copy payload between '<' and that comma
  char tmp[128];
  size_t n = 0;
  const char* p = start + 1;
  while (p < cs && n < sizeof(tmp) - 1) {
    tmp[n++] = *p++;
  }
  tmp[n] = 0;

  char pump_tmp = 'a';
  int parsed = sscanf(tmp, "%d,%d,%d,%d,%d,%c",
                      &port_cmd, &stbd_cmd, &estop, &auto_send, &kill_send, &pump_tmp);

  if (parsed != 6) return false;

  if (pump_tmp != 'A' && pump_tmp != 'a') pump_tmp = 'a';
  pump_char = pump_tmp;

  return true;
}

// =========================================================
//                 APPLY COMMAND
// =========================================================
void apply_command(int port_cmd,
                   int stbd_cmd,
                   int estop,
                   int auto_send,
                   int kill_send,
                   char pump_char)
{
  port_cmd = constrain(port_cmd, PWM_MIN, PWM_MAX);
  stbd_cmd = constrain(stbd_cmd, PWM_MIN, PWM_MAX);

  ROS_PortThrust = (int16_t)port_cmd;
  ROS_StbdThrust = (int16_t)stbd_cmd;

  isAutoFlag   = (auto_send != 0);
  isKilledFlag = (kill_send != 0);

  pump_cmd = (pump_char == 'A') ? 'A' : 'a';

  if (wasD_Estop || isKilledFlag || (estop != 0)) {
    ROS_PortThrust = PWM_NEUTRAL;
    ROS_StbdThrust = PWM_NEUTRAL;
    pump_cmd = 'a';
  }
}

// =========================================================
//                 SERVICE JETSON RX
// =========================================================
void service_jetson_rx()
{
  while (Jetson.available() > 0) {

    char c = (char)Jetson.read();
    if (c == '\r') continue;

    if (c == '<') {
      rx_len = 0;
      rx_buf[rx_len++] = c;
      rx_buf[rx_len] = 0;
      continue;
    }

    if (rx_len > 0) {

      if (c == '<') {
        rx_len = 0;
        rx_buf[rx_len++] = '<';
        rx_buf[rx_len] = 0;
        continue;
      }

      if (rx_len < (int)sizeof(rx_buf) - 1) {
        rx_buf[rx_len++] = c;
        rx_buf[rx_len] = 0;
      } else {
        rx_len = 0;
        validChecksum = false;
        continue;
      }

      if (c == '>') {

        int port_cmd = PWM_NEUTRAL;
        int stbd_cmd = PWM_NEUTRAL;
        int estop = 0, auto_send = 1, kill_send = 0;
        char pump_char = 'a';
        uint8_t rx_cc = 0;

        bool ok = parse_jetson_frame(rx_buf,
                                     port_cmd, stbd_cmd,
                                     estop, auto_send, kill_send,
                                     pump_char,
                                     rx_cc);

        if (ok) {
          uint8_t calc = checksum_cmd(port_cmd, stbd_cmd, estop, auto_send, kill_send, pump_char);

          if (calc == rx_cc) {
            validChecksum = true;
            lastJetsonRxMs = millis();
            apply_command(port_cmd, stbd_cmd, estop, auto_send, kill_send, pump_char);
          } else {
            validChecksum = false;
            Jetson.println("[JETSON RX] BAD CHECKSUM");
          }
        } else {
          validChecksum = false;
          Jetson.println("[JETSON RX] PARSE FAIL");
        }

        rx_len = 0;
      }
    }
  }
}

// =========================================================
//                 TELEMETRY HELPERS
// =========================================================
static inline float knots_to_mps(float kts) {
  return kts * 0.514444f;
}

static inline float safeFloat(double v) {
  if (isnan(v) || isinf(v)) return 0.0f;
  return (float)v;
}

// Forward declare checksum (static)
static uint8_t checksum_telemetry(float lat, float lon, float alt, float hdg, float vel);

// =========================================================
//                 SEND TELEMETRY
// =========================================================
void send_jetson_telemetry()
{
  unsigned long now = millis();
  if (now - previousMillis < TELEMETRY_PERIOD_MS) return;
  previousMillis = now;

  const int auto_flag = isAutoFlag ? 1 : 0;
  const int kill_flag = isKilledFlag ? 1 : 0;

  const float batt = systemBattery;
  const int port_pwm_out = port_pwm;
  const int stbd_pwm_out = stbd_pwm;

  const auto& d = garmin.data();
  const bool gps_ok = (d.rmc_valid || d.gga_valid) && (d.fix_quality > 0);

  const float lat = gps_ok ? safeFloat(d.lat_deg) : 0.0f;
  const float lon = gps_ok ? safeFloat(d.lon_deg) : 0.0f;
  const float alt = gps_ok ? safeFloat(d.alt_m)   : 0.0f;
  const float hdg = gps_ok ? safeFloat(d.cog_deg) : 0.0f;
  const float vel = gps_ok ? knots_to_mps(safeFloat(d.sog_knots)) : 0.0f;

  const float ax = accelX;
  const float ay = accelY;
  const float az = accelZ;

  const float qx = quatX;
  const float qy = quatY;
  const float qz = quatZ;
  const float qw = quatW;

  const float wx = gyroX;
  const float wy = gyroY;
  const float wz = gyroZ;

  const float temp_c = temperature;

  uint8_t cc = checksum_telemetry(lat, lon, alt, hdg, vel);

  char msg[256];
  int n = snprintf(
    msg, sizeof(msg),
    "<%d,%d,%.2f,%d,%d,"
    "%.6f,%.6f,%.2f,%.2f,%.2f,"
    "%.3f,%.3f,%.3f,"
    "%.4f,%.4f,%.4f,%.4f,"
    "%.3f,%.3f,%.3f,"
    "%.2f,*%02x>",
    auto_flag, kill_flag, batt,
    port_pwm_out, stbd_pwm_out,
    lat, lon, alt,
    hdg, vel,
    ax, ay, az,
    qx, qy, qz, qw,
    wx, wy, wz,
    temp_c,
    cc
  );

  if (n > 0 && n < (int)sizeof(msg)) {
    Jetson.println(msg);
  }
}

// =========================================================
//                 TELEMETRY CHECKSUM
// =========================================================
static uint8_t checksum_telemetry(float lat, float lon, float alt, float hdg, float vel)
{
  const int auto_flag = isAutoFlag ? 1 : 0;
  const int kill_flag = isKilledFlag ? 1 : 0;

  const float batt = systemBattery;

  const int port_pwm_out = port_pwm;
  const int stbd_pwm_out = stbd_pwm;

  const float ax = accelX;
  const float ay = accelY;
  const float az = accelZ;

  const float qx = quatX;
  const float qy = quatY;
  const float qz = quatZ;
  const float qw = quatW;

  const float wx = gyroX;
  const float wy = gyroY;
  const float wz = gyroZ;

  const float temp_c = temperature;

  float sum =
    (float)auto_flag + (float)kill_flag + batt +
    (float)port_pwm_out + (float)stbd_pwm_out +
    lat + lon + alt + hdg + vel +
    ax + ay + az +
    qx + qy + qz + qw +
    wx + wy + wz +
    temp_c;

  int checksum = (int)sum;
  if (checksum < 0) checksum = -checksum;
  return (uint8_t)(checksum % 256);
}

// -----------------------------
// Init helpers (called by .ino)
// -----------------------------

void RB26_hw_init()
{
  // Solid state relay control
  pinMode(T500_EN, OUTPUT);
  digitalWrite(T500_EN, HIGH); // disable thrusters at start

  // Fan
  pinMode(FAN_EN, OUTPUT);
  digitalWrite(FAN_EN, LOW);

  // Battery sense
  pinMode(BATT_V, INPUT);
  pinMode(BATT_SENS_EN, OUTPUT);
  digitalWrite(BATT_SENS_EN, LOW);

  // Light tower
  pinMode(LT_ORANGE, OUTPUT); digitalWrite(LT_ORANGE, LOW);
  pinMode(LT_GREEN,  OUTPUT); digitalWrite(LT_GREEN,  LOW);
  pinMode(LT_RED,    OUTPUT); digitalWrite(LT_RED,    LOW);

  // Safety sense
  pinMode(ESTOP_SENSE, INPUT);

  // RC controller pins
  pinMode(CH1, INPUT);
  pinMode(CH2, INPUT);
#ifdef CH3
  pinMode(CH3, INPUT);
#endif
  pinMode(CH4, INPUT);
  pinMode(CH5, INPUT);
  pinMode(SBUS_CH6, INPUT);

  // Temp
  pinMode(TEMP, INPUT);
  analogReadResolution(12);
}

void RB26_gps_init(HardwareSerial &gpsSerial, uint32_t baud)
{
  gpsSerial.begin(baud);      // e.g. Serial6 @ 38400
  garmin.begin(gpsSerial);
  garmin.setRawLineCallback(onNmeaLine);
}

void RB26_pump_init(uint32_t baud)
{
  Serial3.begin(baud);        // pump controller
  pump_cmd = 'a';
  Serial3.print("a");         // OFF
}

void RB26_servos_init()
{
  portThrust.attach(PORT_PW);
  stbdThrust.attach(STBD_PW);
  portThrust.writeMicroseconds(PWM_NEUTRAL);
  stbdThrust.writeMicroseconds(PWM_NEUTRAL);
}

bool RB26_imu_init()
{
  return imuReader.begin();
}

void RB26_rc_init()
{
  READ_RC_begin();
}

// Optional (recommended) wrappers so .ino doesn't touch objects directly
void RB26_update_gps() { garmin.update(); }
void RB26_update_imu() { imuReader.update(); }

