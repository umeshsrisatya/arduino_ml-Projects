import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import warnings
import serial
warnings.filterwarnings('ignore')

dataset=pd.read_csv(r"C:\Users\happy\OneDrive\Desktop\aditiya__college\aditiya college\temp_heart\hb_temp_data.csv")

print(dataset.head())

print(dataset.shape)

print(dataset.describe())

print(dataset.isna().sum())

x=dataset.drop('result',axis=1)
y=dataset['result']

print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

print(x.shape,x_train.shape,y_train.shape)

model=RandomForestClassifier()
model.fit(x_train,y_train)

x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy score for RFC train data',training_data_accuracy*100)

x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy score for RFC test data',training_data_accuracy*100)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)

x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy score for DTC train data',training_data_accuracy*100)

x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy score for DTC test data',training_data_accuracy*100)

model=KNeighborsClassifier()
model.fit(x_train,y_train)
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy score for KNC train data',training_data_accuracy*100)
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy score for KNC test data',training_data_accuracy*100)

model=svm.SVC(kernel='linear')
model.fit(x_train,y_train)

x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy score for SVM train data',training_data_accuracy*100)

x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy score for SVM test data',training_data_accuracy*100)

# Define Classifier names and their corresponding test accuracies
classifier_names=['RFC','DTC','KNN','SVM']
train_accuracies=[99.0,99.0,93.9,99.0]
test_accuracies=[99.0,99.0,93.9,99.0]

# Plotting data using matplotlib
plt.figure(figsize=(10,6))
width=0.35
x=range(len(classifier_names))


plt.bar(x,train_accuracies,width,label='Training Accuracy',alpha=0.7)
plt.bar([i+width for i in x],test_accuracies,width,label='Testing Accuracy',alpha=0.7)
plt.xlabel('Classifiers')
plt.ylabel('Accuracy(%)')
plt.title('Classifier Training and Testing Accuracy Comparision')
plt.xticks([i+width/2 for i in x],classifier_names)
plt.ylim(0,100) # Set the y-axis limit to 0-100%
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()

#Countplot for 'result' distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='result', data=dataset)
plt.title('Countplot of Result')
plt.show()


try:
    ser = serial.Serial('COM4', 9600)
    new_data = []  # Initialize a list to store sensor data
    while True:
       #print('STT')
       data = ser.readline().decode().strip()
       sensor_values = data.split(',')
    # Extract sensor values
       heart_beat = float(sensor_values[0])
       body_temperature = float(sensor_values[1])
       print('heart_beat', heart_beat)
       print('body_temperature', body_temperature)
       new_data.append([heart_beat, body_temperature])

       if len(new_data) >= 0:
            predictions = model.predict(np.array(new_data))
            result = predictions[-1]  # Check the correct index for predictions
            print(result)

            if result == 0:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())
                
                print('Body Temperature and Heart Beat is Good')
                print()
            elif result == 1: 
                data_sending = 0
                data_sending = str(data_sending)
                ser.write(data_sending.encode())  
                print('Heart Beat is Normal and Body Temperature is High. \n This scenario might suggest an imbalance or a response to an underlying condition, infection, or external factors affecting the body thermoregulation,Hormonal Changes.')
                print()
            elif result == 2:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())
                print(' Heart Beat is High and Body Temperature is Normal\n This scenario could be due to Physical Activity,Anxiety,Stimulants,Dehydration ')
                print()
            elif result == 3:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())   
                print(' Heart Beat is High and Body Temperature is High\nIt could indicate various conditions that might be causing physiological stress on the body:')
                print()
            elif result == 4:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())   
                print('Heart Beat is Low and Body Temperature is Low\n if accompanied by symptoms like shivering, confusion, weakness, pale skin, or lethargy, its essential to seek medical attention promptly.')
                print()
            elif result == 5:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())   
                print('Heart Beat is Low and Body Temperature is Normal\n This scenario could be due to Athletic Conditioning,Medication Side Effects,Healthy Individuals,Vagal Tone.')
                print()
            elif result == 6:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())   
                print('Heart Beat is Normal and Body Temperature is Low\n it could indicate various conditions or situations that result in a decreased body temperature, also known as hypothermia.')
                print()
            elif result == 7:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())   
                print('Heart Beat is High and Body Temperature is Low\n if accompanied by symptoms such as confusion, weakness, paleness, shivering, or signs of shock, its crucial to seek immediate medical attention.')
                print()
            elif result == 8:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())   
                print('Heart Beat is Low and Body Temperature is High\n if accompanied by symptoms like confusion, weakness, dizziness, fainting, shortness of breath, chest pain, or other concerning symptoms, seeking immediate medical attention is essential.')
                print()
            
            new_data = []
       else:
            print('Please check the sensor')
            print('_' * 80)
except ValueError:
    print("Invalid input or you chose to exit. Exiting the loop.")
