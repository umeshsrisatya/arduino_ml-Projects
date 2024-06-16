const int tdsSensorPin = A0; // Analog pin for TDS sensor
const int buzzer = 13;

void setup() {
  Serial.begin(9600);
  pinMode(buzzer, OUTPUT);
  digitalWrite(buzzer, LOW);
}

void loop() {
  float tdsValue = readTDS(); // Read TDS value from the sensor
  Serial.println(tdsValue);

  // Check TDS value and activate the buzzer if it exceeds a threshold
  if (tdsValue > 500) { // Adjust the threshold as needed
    digitalWrite(buzzer, HIGH);
  } else {
    digitalWrite(buzzer, LOW);
  }

  delay(1000); // Adjust the delay as needed
}

float readTDS() {
  int sensorValue = analogRead(tdsSensorPin);
  // You may need to calibrate the sensor to obtain accurate TDS values
  // Refer to the sensor documentation for calibration details
  float tdsValue = map(sensorValue, 0, 1023, 0, 1000); // Adjust the mapping based on your sensor's characteristics
  return tdsValue;
}
