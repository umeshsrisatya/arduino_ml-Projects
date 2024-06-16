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

dataset=pd.read_csv(r"C:\placement\aditiya__college\aditiya college\co_mq135\mq7_mq135_data.csv")

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
print('Accuracy score for RFC train data',training_data_accuracy*100)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)

x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy score for DTC train data',training_data_accuracy*100)

x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy score for DTC train data',training_data_accuracy*100)

model=KNeighborsClassifier()
model.fit(x_train,y_train)
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy score for KNC train data',training_data_accuracy*100)
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy score for KNC train data',training_data_accuracy*100)

model=svm.SVC(kernel='linear')
model.fit(x_train,y_train)

x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy score for SVM train data',training_data_accuracy*100)

x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy score for SVM train data',training_data_accuracy*100)

# Define Classifier names and their corresponding test accuracies
classifier_names=['RFC','DTC','KNN','SVM']
train_accuracies=[100.0,100.0,93.9,100.0]
test_accuracies=[100.0,100.0,93.9,100.0]

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

# Pairplot graphs
sns.pairplot(dataset, hue='result')
plt.show()


try:
    ser = serial.Serial('COM4', 9600)
    new_data = []  # Initialize a list to store sensor data
    while True:
       data = ser.readline().decode().strip()
       sensor_values = data.split(',')
       
    # Extract sensor values
       carbon_monoxide = float(sensor_values[0])
       air = float(sensor_values[1])
       print('Carbon Monoxide', carbon_monoxide)
       print('Air Quality: ', air)
       new_data.append([carbon_monoxide, air])
       #print('ss')

       if len(new_data) >= 0:
            predictions = model.predict(np.array(new_data))
            result = predictions[-1]  # Check the correct index for predictions
            print(result)

            if result == 0:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())
                
                print('Carbon Monoxide and Carbon Monoxide COnditions are Good')
                print()
            elif result == 1: 
                data_sending = 0
                data_sending = str(data_sending)
                ser.write(data_sending.encode())  
                print('Carbon Monoxide Condition  is Normal and Air Quality Condition is High. \n  which is beneficial for respiratory health and overall well-being.')
                print()
            elif result == 2:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())
                print('  Carbon Monoxide Condition is High and Air Quality Condition is Normal\n This might imply that other pollutants apart from CO are within acceptable limits and not significantly impacting the overall air quality index. ')
                print()
            elif result == 3:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())   
                print(' Carbon Monoxide Condition is High and Air Quality Condition is High\n This might imply that other pollutants apart from CO are within acceptable limits and not significantly impacting the overall air quality index.')
                print()
            elif result == 4:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())   
                print('Carbon Monoxide Condition is Low and Air Quality Condition is Low\nIt leading to issues such as smog formation, environmental degradation, and harm to plants and wildlife. ')
                print()
            elif result == 5:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())   
                print('Carbon Monoxide Condition is Low and Air Quality Condition is Normal\n These efforts might involve reducing emissions from vehicles, industries, and promoting cleaner energy sources to sustain the current air quality index.')
                print()
            elif result == 6:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())   
                print('Carbon Monoxide Condition is Normal and Air Quality Condition is Low\n This might involve reducing emissions, implementing stricter regulations, promoting cleaner technologies, and increasing public awareness of air quality-related issues.')
                print()
            elif result == 7:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())   
                print('Carbon Monoxide Condition is High and Air Quality Condition is Low\n  This includes pollutants such as particulate matter, nitrogen dioxide, sulfur dioxide, volatile organic compounds, and others.')
                print()
            elif result == 8:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())   
                print('Carbon Monoxide Condition is Low and Air Quality is High\nThis is beneficial for respiratory health, suggesting a reduced risk of CO-related health issues. ')
                print()
            
            new_data = []
       else:
            print('Please check the sensor')
            print('_' * 80)
except ValueError:
    print("Invalid input or you chose to exit. Exiting the loop.")
