#pragma once
#include <Arduino.h>
#include <math.h>

class GarminNMEA {
public:
  struct GpsData {
    // RMC
    bool   rmc_valid = false;
    double lat_deg = NAN;
    double lon_deg = NAN;
    double sog_knots = NAN;
    double cog_deg = NAN;
    uint32_t date_ddmmyy = 0;   // ddmmyy
    uint32_t time_hhmmss = 0;   // hhmmss (integer seconds)

    // GGA
    bool   gga_valid = false;
    uint8_t fix_quality = 0;    // 0 invalid, 1 GPS, 2 DGPS...
    uint8_t sats_used = 0;
    double hdop = NAN;
    double alt_m = NAN;

    // GSA
    bool   gsa_valid = false;
    uint8_t fix_type = 0;       // 1 none, 2 2D, 3 3D
    double pdop = NAN;
    double vdop = NAN;

    // GSV
    bool   gsv_valid = false;
    uint8_t sats_in_view = 0;

    // VTG
    bool   vtg_valid = false;
    double speed_kmh = NAN;

    // PGRME
    bool   pgrme_valid = false;
    double epe_horz_m = NAN;
    double epe_vert_m = NAN;
    double epe_sphr_m = NAN;

    uint32_t last_sentence_ms = 0;
  };

  using RawLineCallback = void (*)(const char* line);

  GarminNMEA();
  void begin(Stream& gpsStream);
  void update();
  void setRawLineCallback(RawLineCallback cb);
  const GpsData& data() const;

  void sendCommand(const char* payload);
  void enableSentence(const char* sentence);
  void disableSentence(const char* sentence);
  void enableAll();
  void disableAll();

private:
  static constexpr size_t NMEA_MAX = 160;

  Stream* _s = nullptr;
  RawLineCallback _rawCb = nullptr;

  char _buf[NMEA_MAX];
  size_t _len = 0;

  GpsData _gps;

  void handleLine(char* line);
  static void trimCRLF(char* s);
  static int splitCsv(char* s, char* tok[], int maxTok);

  static uint8_t hexNibble(char c);
  static bool checksumOk(const char* line);
  static void computeChecksum(const char* payload, char outCS[3]);

  static double parseLatLonDeg(const char* ddmm, char hemi);

  void parseRMC(char* line);
  void parseGGA(char* line);
  void parseGSA(char* line);
  void parseGSV(char* line);
  void parseVTG(char* line);
  void parsePGRME(char* line);
};
