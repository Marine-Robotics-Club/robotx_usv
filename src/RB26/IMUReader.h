#ifndef IMUREADER_H
#define IMUREADER_H

#include <Adafruit_BNO055.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <utility/imumaths.h>

// Declare externally defined variables (defined in .ino)
extern volatile float accelX, accelY, accelZ;
extern volatile float gyroX, gyroY, gyroZ;
extern volatile float magX, magY, magZ;
extern volatile float eulerX, eulerY, eulerZ;
extern volatile float quatW, quatX, quatY, quatZ;
extern volatile float linX, linY, linZ;
extern volatile float gravX, gravY, gravZ;

class IMUReader {
public:
    IMUReader();
    bool begin();
    void update();
private:
    Adafruit_BNO055 bno;
};

#endif
