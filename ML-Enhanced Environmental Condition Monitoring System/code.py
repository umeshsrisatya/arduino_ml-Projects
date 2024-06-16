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

dataset=pd.read_csv(r"C:\Users\happy\OneDrive\Desktop\aditiya__college\aditiya college\temp_rain_light\temp_rain_light_data.csv")

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

model=LogisticRegression()
model.fit(x_train,y_train)

x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy score for LOGR train data',training_data_accuracy*100)

x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy score for LOGR train data',training_data_accuracy*100)


# Visualize target variable distribution (Histogram)
plt.figure(figsize=(8, 6))
sns.histplot(y)
plt.title('Target Variable Distribution')
plt.xlabel('Result')
plt.ylabel('Count')
plt.show()

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


try:
    ser = serial.Serial('COM4', 9600)
    new_data = []  # Initialize a list to store sensor data
    while True:
       data = ser.readline().decode().strip()
       sensor_values = data.split(',')
    # Extract sensor values
       temperature_value = float(sensor_values[0])
       rain_value = float(sensor_values[1])
       light_value = float(sensor_values[2])

       print('temperature_value', temperature_value)
       print('rain_value', rain_value)
       print('light_value',light_value )

       new_data.append([temperature_value,rain_value, light_value])

       if len(new_data) >= 0:  
            
            predictions = model.predict(np.array(new_data))
            result = predictions[-1]  # Check the correct index for predictions
            print(result)
            if result == 0:
                print('All conditions are Normal.')
                print()
            elif result == 1: 
                print('Rain, Temperature Conditions are Normal and light Condition is High.\n This could indicate that there are minimal clouds obstructing the sunlight, leading to a bright and sunny day.')
                print()
            elif result == 2: 
                print('Light, Temperature  Conditions are Normal and Rain Condition is High \nThis could imply that atmospheric conditions such as increased humidity')
                print()
            elif result == 3:
                print('Rain,Light Conditions are High and Temperature Condition is Normal\n This scenario could occur during periods of unsettled weather where showers or thunderstorms develop but are interspersed with breaks of sunshine.')
                print()
            elif result == 4: 
                print('Rain,Light Conditions are Normal and Temperature Condition is High \n This scenario might be typical during periods of summer or in regions with a hot climate where sunny conditions ')
                print()
            elif result == 5: 
                print('Light, Temperature  Conditions are High and Rain Condition is Normal \n This scenario is typical during periods of stable high-pressure systems or in regions with arid climates where rainfall is infrequent.')
                print()
            elif result == 6:
                print('Rain, Temperature Conditions are High and Light Condition is Normal \n This scenario is often associated with weather systems such as fronts or tropical disturbances that bring significant rainfall')
                print()
            elif result == 7: 
                print('Rain, Light ,Temperature Condition are High\n This condition is impossible in Nature')
                print()
            elif result == 8: 
                print('Rain, Temperature Conditions are High and Light Condition is Low \n A low light condition implies reduced visibility or limited sunlight.')
                print()
            elif result == 9:
                print('Light, Temperature Conditions are High and Rain Condition is Low \nThis scenario is typical during periods of stable high-pressure systems or in regions with arid climates where rainfall is infrequent.')
                print()
            elif result == 10: 
                print('Rain ,Light Conditions are Low and Temperature Condition is High\n In this scenario Atmosphere is not Exist ')
                print()
            elif result == 11: 
                print('Rain, Light Conditions are High and Temperature Condition is Low\n This condition is impossible in Nature ')
                print()
            elif result == 12:
                print('Light,Temperature  Conditions are Low and Rain Condition is High \n This may lead to continuous drizzle or light rain rather than heavy rainfall')
                print()
            elif result == 13: 
                print('Rain, Temperature  Conditions are Low and Light Condition is High \n The air might feel dry, and the lack of moisture can contribute to a heightened sensation of heat, potentially causing discomfort. ')
                print()
            elif result == 14: 
                print('Rain,Light, Temperature  Conditions are Low\nThis  Condition is Unpredictable')
                print()
            elif result == 15:
                print('Rain, Temperature  Conditions are Normal and Light Condition is Low \n this conditions typically create a cool and damp environment, the air might feel moist, and surfaces could remain wet due to the normal precipitation. ')
                print()
            elif result == 16: 
                print('Light, Temperature  Conditions are Normal and Rain Condition is Low \n This indicates that there is neither excessive cloud cover nor exceptionally clear skies. Its a typical level of daylight for the time of day and year.')
                print()
            elif result == 17: 
                print('light, Rain Conditions are Low and Temperature Condition is Normal \n Condion is impossible ')
                print()
            elif result == 18:
                print('Rain,Light  Conditions are Normal and Temperature Condition is Low\n  This  Condition is Unpredictable')
                print()
            elif result == 19: 
                print('Light, Temperature  Conditions are Low and Rain Condition is Normal \n This scenario might occur during transitional seasons or in regions where weather patterns are variable, leading to occasional rainfall even during periods of lower temperatures and light.')
                print()
            elif result == 20: 
                print('Rain, Temperature  Conditions are Low and Light Condition is Normal \n This could indicate colder weather conditions.')
                print()
            elif result == 21:
                print('Temperature Condition is Normal, Rain Condition is High and Light  Condition is Low \nThe atmosphere likely contains ample moisture, and there may be factors such as low pressure systems or atmospheric instability contributing to the likelihood of precipitation.')
                print()
            elif result == 22: 
                print('Temperature Condition is Normal, Rain Condition is Low and Light  Condition is High \n This scenario might occur during hot and humid conditions, where occasional showers or storms can develop despite the reduced light.')
                print()
            elif result == 23: 
                print('Rain Condition is Normal, Light Condition is Low and Temperature  Condition is High \n  This could indicate colder weather conditions.')
                print()
            elif result == 24:
                print('Temperature Condition is Low,  Light Condition is High and Rain Condition is Normal\n This scenario might occur during colder seasons or in regions where precipitation is common despite the lower temperatures and normal light conditions.')
                print()
            elif result == 25: 
                print('Temperature Condition is Low, Light Condition is Normal and  Rain Condition is Normal \n The air might feel relatively heavy with moisture, even without significant rainfall.')
                print()
            elif result == 26: 
                print('Temperature Condition is High, Rain Condition is Low and Light Condition is Normal \n This scenario might result in moderate to heavy rainfall despite the lower moisture content in the air')
                print()
    
            new_data = []
       else:
            print('Please check the sensor')
            print('_' * 80)
except ValueError:
    print("Invalid input or you chose to exit. Exiting the loop.")
