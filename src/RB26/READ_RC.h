#pragma once
#include <Arduino.h>

/*
  READ_RC.h
  ----------
  Teensy 4.1 RC PWM reader (mixed polarity)

  CH1–CH5 : inverted (active LOW pulse)
  CH6     : non-inverted (active HIGH pulse)

  User must define:
    CH1, CH2, CH3, CH4, CH5, SBUS_CH6
*/

// ===== Public RC outputs (microseconds) =====
extern volatile uint16_t RC_CH1;
extern volatile uint16_t RC_CH2;
extern volatile uint16_t RC_CH3;
extern volatile uint16_t RC_CH4;
extern volatile uint16_t RC_CH5;
extern volatile uint16_t RC_CH6;

// ===== Internal timing =====
static volatile uint32_t _tStart1, _tStart2, _tStart3, _tStart4, _tStart5, _tStart6;

// ===== ISR handlers =====
// CH1–CH5 inverted (LOW pulse)
static void isrCH1() {
  bool lvl = digitalReadFast(CH1);
  uint32_t now = micros();
  if (!lvl) _tStart1 = now;
  else      RC_CH1 = (uint16_t)(now - _tStart1);
}

static void isrCH2() {
  bool lvl = digitalReadFast(CH2);
  uint32_t now = micros();
  if (!lvl) _tStart2 = now;
  else      RC_CH2 = (uint16_t)(now - _tStart2);
}

static void isrCH3() {
  bool lvl = digitalReadFast(CH3);
  uint32_t now = micros();
  if (!lvl) _tStart3 = now;
  else      RC_CH3 = (uint16_t)(now - _tStart3);
}

static void isrCH4() {
  bool lvl = digitalReadFast(CH4);
  uint32_t now = micros();
  if (!lvl) _tStart4 = now;
  else      RC_CH4 = (uint16_t)(now - _tStart4);
}

static void isrCH5() {
  bool lvl = digitalReadFast(CH5);
  uint32_t now = micros();
  if (!lvl) _tStart5 = now;
  else      RC_CH5 = (uint16_t)(now - _tStart5);
}

// CH6 non-inverted (HIGH pulse)
static void isrCH6() {
  bool lvl = digitalReadFast(SBUS_CH6);
  uint32_t now = micros();
  if (lvl)  _tStart6 = now;
  else      RC_CH6 = (uint16_t)(now - _tStart6);
}

// ===== Initialization =====
static inline void READ_RC_begin() {
  pinMode(CH1, INPUT);
  pinMode(CH2, INPUT);
  pinMode(CH3, INPUT);
  pinMode(CH4, INPUT);
  pinMode(CH5, INPUT);
  pinMode(SBUS_CH6, INPUT);

  attachInterrupt(digitalPinToInterrupt(CH1), isrCH1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(CH2), isrCH2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(CH3), isrCH3, CHANGE);
  attachInterrupt(digitalPinToInterrupt(CH4), isrCH4, CHANGE);
  attachInterrupt(digitalPinToInterrupt(CH5), isrCH5, CHANGE);
  attachInterrupt(digitalPinToInterrupt(SBUS_CH6), isrCH6, CHANGE);
}

// ===== Safe read helper =====
static inline uint16_t READ_RC(uint8_t ch) {
  uint16_t v;

  noInterrupts();
  switch (ch) {
    case 1: v = RC_CH1; break;
    case 2: v = RC_CH2; break;
    case 3: v = RC_CH3; break;
    case 4: v = RC_CH4; break;
    case 5: v = RC_CH5; break;
    case 6: v = RC_CH6; break;
    default: v = 0;     break;
  }
  interrupts();

  // RC sanity limits
  if (v < 800 || v > 2200) return 0;
  return v;
}
