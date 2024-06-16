#include <OneWire.h>
#include <DallasTemperature.h>
#define ONE_WIRE_BUS A0
const int hbs = A2;
const int buzzer = 13;

int heartrate;

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);


void setup() {
  Serial.begin(9600);
  sensors.begin();  // Start up the library
  pinMode(hbs,INPUT);
  pinMode(buzzer, OUTPUT);
  digitalWrite(buzzer, LOW);
 }

void loop() {
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
  get_HB_data();
  Serial.print(",");
  Serial.print((sensors.getTempCByIndex(0) * 9.0) / 5.0 + 32.0);
  Serial.println();
  delay(500);
}

void get_HB_data()
{
int cnt=0;
double tempv=millis();
   while(millis()<(tempv+7000))
   {
     if((digitalRead(hbs)) == LOW)
     {
     cnt++; 
     delay(400);
     }
   }
cnt=cnt*7;
heartrate=cnt;
cnt=0;
Serial.write(heartrate/100+48);
Serial.write((heartrate/10)%10+48);
Serial.write((heartrate/1)%10+48);
}
