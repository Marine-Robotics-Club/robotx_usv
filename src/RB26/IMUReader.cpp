#include "IMUReader.h"

// Define constructor
IMUReader::IMUReader() : bno(55, 0x28, &Wire) {}

bool IMUReader::begin() {
    if (!bno.begin()) return false;
    bno.setExtCrystalUse(true);
    return true;
}

void IMUReader::update() {
    sensors_event_t accel, gyro, mag;

    imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
    imu::Quaternion quat = bno.getQuat();
    imu::Vector<3> lin = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
    imu::Vector<3> grav = bno.getVector(Adafruit_BNO055::VECTOR_GRAVITY);

    bno.getEvent(&accel, Adafruit_BNO055::VECTOR_ACCELEROMETER);
    bno.getEvent(&gyro, Adafruit_BNO055::VECTOR_GYROSCOPE);
    bno.getEvent(&mag, Adafruit_BNO055::VECTOR_MAGNETOMETER);

    // Save to global variables
    accelX = accel.acceleration.x;
    accelY = accel.acceleration.y;
    accelZ = accel.acceleration.z;

    gyroX = gyro.gyro.x;
    gyroY = gyro.gyro.y;
    gyroZ = gyro.gyro.z;

    magX = mag.magnetic.x;
    magY = mag.magnetic.y;
    magZ = mag.magnetic.z;

    eulerX = euler.x();
    eulerY = euler.y();
    eulerZ = euler.z();

    quatW = quat.w();
    quatX = quat.x();
    quatY = quat.y();
    quatZ = quat.z();

    linX = lin.x();
    linY = lin.y();
    linZ = lin.z();

    gravX = grav.x();
    gravY = grav.y();
    gravZ = grav.z();
}
