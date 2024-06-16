const int buzzer = 13;

void setup() {
  Serial.begin(9600);
  pinMode(buzzer, OUTPUT);
  digitalWrite(buzzer, LOW);
 }

void loop() {
  char receivedChar;
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
  float co_value = analogRead(A2);
  co_value = co_value/2;
  float mq135_value = analogRead(A1);
 // Send sensor data to Raspberry Pi
  Serial.print(co_value);
  Serial.print(",");
  Serial.print(mq135_value);
  Serial.println();
  delay(500);
}
