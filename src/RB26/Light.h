// Simple class to manage light tower lights
// the update() method should be called
// periodically (100ms) for blinking to function
// correctly.


#ifndef LIGHT_H
#define LIGHT_H

#define L_OFF         0
#define L_ON          1
#define L_SLOW_BLINK  2
#define L_FAST_BLINK  3

class Light
{
  private:
    int _pin;
    int _lightState = L_OFF;
    int _lightCnt = 0;    

  public:
    Light(int pin);
    void setState(int lightState);
    void update();
};

Light::Light(int pin)
{
  _pin = pin;
  pinMode(_pin, OUTPUT);
}

/*** public functions ***/
void Light::setState(int lightState)
{
  _lightState = lightState;
}

void Light::update()
{
    switch(_lightState)
    {
      case L_OFF:
        digitalWrite(_pin, LOW);
        _lightCnt = 0;
        break;
      case L_ON:
        digitalWrite(_pin, HIGH);
        _lightCnt = 0;
        break;
      case L_SLOW_BLINK:
        _lightCnt++;
        if (_lightCnt > 9)
        {
          digitalWrite(_pin,!digitalRead(_pin));
          _lightCnt = 0;
        }
        break;        
      case L_FAST_BLINK:
        digitalWrite(_pin,!digitalRead(_pin));
        _lightCnt = 0;
        break;
    }  
}

#endif
