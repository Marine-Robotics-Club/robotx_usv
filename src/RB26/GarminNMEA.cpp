#include "GarminNMEA.h"
#include <ctype.h>
#include <string.h>

GarminNMEA::GarminNMEA() {}

void GarminNMEA::begin(Stream& gpsStream) {
  _s = &gpsStream;
  _len = 0;
}

void GarminNMEA::setRawLineCallback(RawLineCallback cb) {
  _rawCb = cb;
}

const GarminNMEA::GpsData& GarminNMEA::data() const {
  return _gps;
}

void GarminNMEA::update() {
  if (!_s) return;

  while (_s->available()) {
    char c = (char)_s->read();
    if (c == '\n') {
      _buf[_len] = '\0';
      if (_len > 6) handleLine(_buf);
      _len = 0;
    } else {
      if (_len < NMEA_MAX - 1) _buf[_len++] = c;
      else _len = 0;
    }
  }
}

// ---------------- checksum helpers ----------------
uint8_t GarminNMEA::hexNibble(char c) {
  if (c >= '0' && c <= '9') return (uint8_t)(c - '0');
  if (c >= 'A' && c <= 'F') return (uint8_t)(c - 'A' + 10);
  if (c >= 'a' && c <= 'f') return (uint8_t)(c - 'a' + 10);
  return 0;
}

bool GarminNMEA::checksumOk(const char* line) {
  if (!line || line[0] != '$') return false;

  uint8_t cs = 0;
  const char* p = line + 1;
  while (*p && *p != '*' && *p != '\r' && *p != '\n') cs ^= (uint8_t)(*p++);
  if (*p != '*') return false;

  p++;
  if (!isxdigit((unsigned char)p[0]) || !isxdigit((unsigned char)p[1])) return false;
  uint8_t got = (hexNibble(p[0]) << 4) | hexNibble(p[1]);
  return cs == got;
}

void GarminNMEA::computeChecksum(const char* payload, char outCS[3]) {
  uint8_t cs = 0;
  for (const char* p = payload; *p; ++p) cs ^= (uint8_t)(*p);
  const char* hex = "0123456789ABCDEF";
  outCS[0] = hex[(cs >> 4) & 0xF];
  outCS[1] = hex[cs & 0xF];
  outCS[2] = '\0';
}

// ---------------- command sending ----------------
void GarminNMEA::sendCommand(const char* payload) {
  if (!_s || !payload || !*payload) return;

  char cs[3];
  computeChecksum(payload, cs);

  _s->print('$');
  _s->print(payload);
  _s->print('*');
  _s->print(cs);
  _s->print("\r\n");
}

void GarminNMEA::enableSentence(const char* sentence) {
  if (!sentence || !*sentence) return;
  char payload[64];
  snprintf(payload, sizeof(payload), "PGRMO,%s,1", sentence);
  sendCommand(payload);
}

void GarminNMEA::disableSentence(const char* sentence) {
  if (!sentence || !*sentence) return;
  char payload[64];
  snprintf(payload, sizeof(payload), "PGRMO,%s,0", sentence);
  sendCommand(payload);
}

void GarminNMEA::enableAll() {
  sendCommand("PGRMO,,1");
}

void GarminNMEA::disableAll() {
  sendCommand("PGRMO,,2");
}

// ---------------- parsing helpers ----------------
void GarminNMEA::trimCRLF(char* s) {
  if (!s) return;
  size_t n = strlen(s);
  while (n && (s[n - 1] == '\r' || s[n - 1] == '\n')) {
    s[n - 1] = '\0';
    n--;
  }
}

int GarminNMEA::splitCsv(char* s, char* tok[], int maxTok) {
  int count = 0;
  tok[count++] = s;
  for (char* p = s; *p && count < maxTok; p++) {
    if (*p == ',') {
      *p = '\0';
      tok[count++] = p + 1;
    }
    if (*p == '*') {
      *p = '\0';
      break;
    }
  }
  return count;
}

double GarminNMEA::parseLatLonDeg(const char* ddmm, char hemi) {
  if (!ddmm || !*ddmm) return NAN;

  double v = atof(ddmm);
  double deg = floor(v / 100.0);
  double min = v - deg * 100.0;
  double out = deg + (min / 60.0);
  if (hemi == 'S' || hemi == 'W') out = -out;
  return out;
}

// ---------------- sentence handlers ----------------
void GarminNMEA::handleLine(char* line) {
  trimCRLF(line);
  if (!line || line[0] != '$') return;

  if (_rawCb) _rawCb(line);

  if (!checksumOk(line)) return;
  if (strlen(line) < 6) return;

  char tmp[NMEA_MAX];
  strncpy(tmp, line, sizeof(tmp));
  tmp[sizeof(tmp) - 1] = '\0';

  char id[6] = {0};
  memcpy(id, &tmp[1], 5);

  if      (!strncmp(id, "GPRMC", 5)) parseRMC(tmp);
  else if (!strncmp(id, "GPGGA", 5) || !strncmp(id, "GNGGA", 5)) parseGGA(tmp);
  else if (!strncmp(id, "GPGSA", 5) || !strncmp(id, "GNGSA", 5)) parseGSA(tmp);
  else if (!strncmp(id, "GPGSV", 5) || !strncmp(id, "GNGSV", 5)) parseGSV(tmp);
  else if (!strncmp(id, "GPVTG", 5) || !strncmp(id, "GNVTG", 5)) parseVTG(tmp);
  else if (!strncmp(id, "PGRME", 5)) parsePGRME(tmp);
}

void GarminNMEA::parseRMC(char* line) {
  char* tok[24];
  int n = splitCsv(line, tok, 24);
  if (n < 10) return;

  const char* t = tok[1];
  const char* status = tok[2];
  const char* lat = tok[3];
  const char* latH = tok[4];
  const char* lon = tok[5];
  const char* lonH = tok[6];
  const char* sog = tok[7];
  const char* cog = tok[8];
  const char* date = tok[9];

  _gps.rmc_valid = (status && status[0] == 'A');
  if (t && *t) _gps.time_hhmmss = (uint32_t)atoi(t);
  if (date && *date) _gps.date_ddmmyy = (uint32_t)atoi(date);

  if (lat && latH && *latH) _gps.lat_deg = parseLatLonDeg(lat, latH[0]);
  if (lon && lonH && *lonH) _gps.lon_deg = parseLatLonDeg(lon, lonH[0]);

  if (sog && *sog) _gps.sog_knots = atof(sog);
  if (cog && *cog) _gps.cog_deg = atof(cog);

  _gps.last_sentence_ms = millis();
}

void GarminNMEA::parseGGA(char* line) {
  char* tok[20];
  int n = splitCsv(line, tok, 20);
  if (n < 10) return;

  const char* lat = tok[2];
  const char* latH = tok[3];
  const char* lon = tok[4];
  const char* lonH = tok[5];
  const char* fix = tok[6];
  const char* sats = tok[7];
  const char* hdop = tok[8];
  const char* alt = tok[9];

  _gps.fix_quality = (fix && *fix) ? (uint8_t)atoi(fix) : 0;
  _gps.gga_valid = (_gps.fix_quality > 0);

  _gps.sats_used = (sats && *sats) ? (uint8_t)atoi(sats) : 0;
  _gps.hdop = (hdop && *hdop) ? atof(hdop) : _gps.hdop;
  _gps.alt_m = (alt && *alt) ? atof(alt) : NAN;

  if (lat && latH && *latH) _gps.lat_deg = parseLatLonDeg(lat, latH[0]);
  if (lon && lonH && *lonH) _gps.lon_deg = parseLatLonDeg(lon, lonH[0]);

  _gps.last_sentence_ms = millis();
}

void GarminNMEA::parseGSA(char* line) {
  char* tok[24];
  int n = splitCsv(line, tok, 24);
  if (n < 6) return;

  _gps.gsa_valid = true;
  _gps.fix_type = (tok[2] && *tok[2]) ? (uint8_t)atoi(tok[2]) : 0;

  _gps.pdop = (n >= 3 && tok[n - 3] && *tok[n - 3]) ? atof(tok[n - 3]) : NAN;
  _gps.hdop = (n >= 2 && tok[n - 2] && *tok[n - 2]) ? atof(tok[n - 2]) : _gps.hdop;
  _gps.vdop = (n >= 1 && tok[n - 1] && *tok[n - 1]) ? atof(tok[n - 1]) : NAN;

  _gps.last_sentence_ms = millis();
}

void GarminNMEA::parseGSV(char* line) {
  // $GPGSV,total_msgs,msg_num,sats_in_view,...
  char* tok[24];
  int n = splitCsv(line, tok, 24);
  if (n < 4) return;

  _gps.gsv_valid = true;
  _gps.sats_in_view = (tok[3] && *tok[3]) ? (uint8_t)atoi(tok[3]) : _gps.sats_in_view;
  _gps.last_sentence_ms = millis();
}

void GarminNMEA::parseVTG(char* line) {
  // $GPVTG,cog,T,mag,M,sog,N,kmh,K,mode
  char* tok[16];
  int n = splitCsv(line, tok, 16);
  if (n < 9) return;

  _gps.vtg_valid = true;
  _gps.cog_deg   = (tok[1] && *tok[1]) ? atof(tok[1]) : _gps.cog_deg;
  _gps.sog_knots = (tok[5] && *tok[5]) ? atof(tok[5]) : _gps.sog_knots;
  _gps.speed_kmh = (tok[7] && *tok[7]) ? atof(tok[7]) : NAN;

  _gps.last_sentence_ms = millis();
}

void GarminNMEA::parsePGRME(char* line) {
  // $PGRME,horz,M,vert,M,sph,M
  char* tok[12];
  int n = splitCsv(line, tok, 12);
  if (n < 6) return;

  _gps.pgrme_valid = true;
  _gps.epe_horz_m = (tok[1] && *tok[1]) ? atof(tok[1]) : NAN;
  _gps.epe_vert_m = (tok[3] && *tok[3]) ? atof(tok[3]) : NAN;
  _gps.epe_sphr_m = (tok[5] && *tok[5]) ? atof(tok[5]) : NAN;

  _gps.last_sentence_ms = millis();
}
