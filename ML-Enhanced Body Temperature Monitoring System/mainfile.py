import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor 
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,r2_score
import warnings
import serial    
warnings.filterwarnings('ignore')

dataset=pd.read_csv(r"C:\placement\aditiya__college\aditiya college\body_temperature\temp_data.csv")

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

model = XGBRegressor()
model.fit(x_train,y_train)

x_train_prediction=model.predict(x_train)
training_data_accuracy=r2_score(x_train_prediction,y_train)
print('r2 score for XGB train data',training_data_accuracy*100)

x_test_prediction=model.predict(x_test)
test_data_accuracy=r2_score(x_test_prediction,y_test)
print('r2 score for XGB test data',training_data_accuracy*100)

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

model=LogisticRegression()
model.fit(x_train,y_train)

x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy score for LOGR train data',training_data_accuracy*100)

x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy score for LOGR train data',training_data_accuracy*100)




# Define Classifier names and their corresponding test accuracies
classifier_names=['RFC','XGB','KNN','SVM','LOR']
train_accuracies=[98.9,99.6,98.9,98.7,97.8]
test_accuracies=[98.9,99.6,98.9,98.7,97.8]

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

# Visualize target variable distribution (Histogram)
plt.figure(figsize=(8, 6))
sns.histplot(y)
plt.title('Target Variable Distribution')
plt.xlabel('Result')
plt.ylabel('Count')
plt.show()

try:
    ser = serial.Serial('COM4', 9600)
    new_data = []  # Initialize a list to store sensor data
    
    while True:
        temp_data = ser.readline().decode('utf-8').strip()
        print("Temperature sensor data:", temp_data)
        temp_data = float(temp_data)
        new_data.append([temp_data])  # Convert to float

        if len(new_data) >= 0:
            predictions = model.predict(np.array(new_data))
            result = predictions[-1]  # Check the correct index for predictions
            #print(result)

            if result == 0:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())
                print('At this stage, the body loses heat faster than it can produce it,\nleading to symptoms like shivering, confusion, slurred speech, and a weak pulse. ')
                print()
            elif result == 1: 
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())  
                print('Body temperature may indicate illness or exposure to cold environments.\n It might cause discomfort and sluggishness.')
                print()
            elif result == 2:
                data_sending = 0
                data_sending = str(data_sending)
                ser.write(data_sending.encode())
                print('This is the average body temperature for most people,\n it is considered as standard range.')
                print()
            elif result == 3:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())   
                print('Fever often indicates an immune response to illness.\nIt can cause discomfort, sweating, chills, and other symptoms')
                print()
            elif result == 4:
                data_sending = 1
                data_sending = str(data_sending)
                ser.write(data_sending.encode())
                print('Hyperthermia can result from prolonged exposure to high temperatures or strenuous physical activity in hot environments. \nSymptoms include dizziness, confusion, rapid heart rate, and in severe cases, heatstroke, which is a medical emergency.')
                print()

            new_data = []
        else:
            print('Please check the sensor')
            print('_' * 80)
except ValueError:
    print("Invalid input or you chose to exit. Exiting the loop.")
