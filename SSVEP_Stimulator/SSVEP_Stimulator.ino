#include <Adafruit_NeoPixel.h>

#define PIN 6

// Parameter 1 = number of pixels in strip
// Parameter 2 = Arduino pin number (most are valid)
// Parameter 3 = pixel type flags, add together as needed:
//   NEO_KHZ800  800 KHz bitstream (most NeoPixel products w/WS2812 LEDs)
//   NEO_KHZ400  400 KHz (classic 'v1' (not v2) FLORA pixels, WS2811 drivers)
//   NEO_GRB     Pixels are wired for GRB bitstream (most NeoPixel products)
//   NEO_RGB     Pixels are wired for RGB bitstream (v1 FLORA pixels, not v2)
//   NEO_RGBW    Pixels are wired for RGBW bitstream (NeoPixel RGBW products)
Adafruit_NeoPixel strip = Adafruit_NeoPixel(8, PIN, NEO_GRB + NEO_KHZ800);

class Flasher
{
    // Class Member Variables
    // These are initialized at startup
    int ledPin;               // the number of the LED pin
    unsigned long OnTime;     // milliseconds of on-time
    unsigned long OffTime;    // milliseconds of off-time
    int iRed;
    int iGreen;
    int iBlue;
    int cycleTime = 1000 * 20;
    unsigned long pmCycle;
    float hz;
    // These maintain the current state
    int ledState;                   // ledState used to set the LED
    unsigned long previousMillis;   // will store last time LED was updated
    int started;
    // Constructor - creates a Flasher
    // and initializes the member variables and state
  public:
    Flasher(int pin, long on, long off, int iRed_, int iGreen_, int iBlue_)
    {
      ledPin = pin;

      OnTime = on;
      OffTime = off;

      ledState = LOW;
      previousMillis = 0;
      pmCycle = 0;

      iRed = iRed_;
      iGreen = iGreen_;
      iBlue = iBlue_;
      started = 0;
    }

    Flasher(int pin, long on, long off) : Flasher(pin, on, off, 255, 255, 255) {}
    Flasher(int pin, float hz) : Flasher(pin, hz, 255, 0, 0) {}
    Flasher(int pin, long hz, int iRed_, int iGreen_, int iBlue_) : Flasher(pin, 0, 0, iRed_, iGreen_, iBlue_)
    {
      OnTime = OffTime = 1000 / (2 * hz);
    }

    void Update()
    {
      // check to see if it's time to change the state of the LED
      unsigned long currentMillis = millis();
      /*
      if (currentMillis - pmCycle >= cycleTime || started == 0)
      {
        hz = fmod((hz + 1),7);
        float a = 5 + (hz + 1);
        //hz = fmod(hz + 1, 20);
        //float a = hz + 1;
        OnTime = OffTime = 1000 / (2 * a);
        pmCycle = currentMillis;
        Serial.print(a);
        started = 1;
      }*/

      if ((ledState == HIGH) && (currentMillis - previousMillis >= OnTime))
      {
        ledState = LOW;  // Turn it off
        previousMillis = currentMillis;  // Remember the time
        strip.setPixelColor(ledPin, 0);
        strip.show();
      }
      else if ((ledState == LOW) && (currentMillis - previousMillis >= OffTime))
      {
        ledState = HIGH;  // turn it on
        previousMillis = currentMillis;   // Remember the time
        strip.setPixelColor(ledPin, iRed, iGreen, iBlue);
        strip.show();
      }
    }
};


Flasher leds[] = { {0, 8}, {1, 12}, {2, 17}, {3,10}, {4,1}, {5,1}, {6,1}, {7,12}};

void setup()
{
  strip.begin();
  strip.show(); // Initialize all pixels to 'off'
}

void loop()
{
  leds[0].Update();
  //leds[3].Update();
  //leds[7].Update();
}


