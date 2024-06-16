#include <OneWire.h>
#include <DallasTemperature.h>
#define ONE_WIRE_BUS A1

const int buzzer = 13;

OneWire oneWire(ONE_WIRE_BUS);

DallasTemperature sensors(&oneWire);

void setup(void)
{
  Serial.begin(9600);
  sensors.begin();  // Start up the library
  pinMode(buzzer, OUTPUT);
  digitalWrite(buzzer, LOW);
}

void loop(void)
{
  char receivedChar;
  sensors.requestTemperatures();
  if (Serial.available() > 0) {
    receivedChar = Serial.read();
    //Serial.println(receivedChar);
  }
  if (receivedChar == '1') {
    digitalWrite(buzzer, HIGH);
  }
  if (receivedChar == '0') {
    digitalWrite(buzzer, LOW);
  }
  //print the temperature in Fahrenheit
  Serial.print((sensors.getTempCByIndex(0) * 9.0) / 5.0 + 32.0);
  Serial.println();
  delay(500);
}
