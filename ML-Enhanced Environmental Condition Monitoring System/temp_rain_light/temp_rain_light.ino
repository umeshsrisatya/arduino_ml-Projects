#include<DHT.h>
#define DHT_TYPE DHT11

#define DHT_PIN A1
DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  dht.begin();
  Serial.begin(9600);
}

void loop() {
  // Read sensor values (replace these with your actual sensor readings)
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  float rainValue = analogRead(A2);
  float lightValue = analogRead(A3);

  // Send sensor data to Raspberry Pi
  Serial.print(temperature);
  Serial.print(",");
  Serial.print(rainValue);
  Serial.print(",");
  Serial.print(lightValue);
  Serial.println();
  delay(500);  // Adjust delay based on your requirements
}
