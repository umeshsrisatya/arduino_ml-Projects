import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import warnings
import serial
warnings.filterwarnings('ignore')

dataset=pd.read_csv(r"C:\placement\aditiya__college\aditiya college\light_soil_temp\temp_soil_light_data.csv")

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
       #print('STT')
       data = ser.readline().decode().strip()
       sensor_values = data.split(',')
    # Extract sensor values
       Temperature = float(sensor_values[0])
       Soil = float(sensor_values[1])
       Light = float(sensor_values[2])

       print('Temperature:', Temperature)
       print('Soil :', Soil)
       print('Light :', Light)

       new_data.append([Temperature,Soil, Light])

       if len(new_data) >= 0:  
            
            predictions = model.predict(np.array(new_data))
            result = predictions[-1]  # Check the correct index for predictions
            #print(result)
            if result == 0:
                print('All conditions are Normal.')
                print()
            elif result == 1: 
                print('Temperature, Soil Conditions are Normal and Light Condition is High.\n It suggests that the temperature and soil moisture levels are within suitable ranges to support healthy plant growth.')
                print()
            elif result == 2: 
                print('Temperature, Light Conditions are Normal and Soil Condition is High.\n This creates an ideal environment for robust plant growth, allowing for improved crop yields, healthy vegetation, and thriving ecosystems. ')
                print()
            elif result == 3:
                print('Soil, Light Conditions are High and Temperature Condition is Low.\n This can promote healthy root development, nutrient availability, and overall plant growth.')
                print()
            elif result == 4: 
                print('Soil, Light Conditions are Normal and Temperature Condition is High.\n This supports healthy root development, nutrient availability, and overall plant growth.')
                print()
            elif result == 5: 
                print('Temperature, Light Conditions are High and Soil Condition is Normal.\nThis promotes healthy root development, nutrient availability, and overall plant growth, creating a suitable environment for plant root systems. ')
                print()
            elif result == 6:
                print('Temperature, Soil Conditions are High and Light Condition is Normal.\n This might enhance soil health and fertility.')
                print()
            elif result == 7: 
                print('Temperature, Soil, Light Conditions are High.\n It leading to increased biomass production, quick maturation of crops, and healthy vegetation.')
                print()
            elif result == 8: 
                print('Temperature, Soil Conditions are High and Light Condition is Low.\nSome plant species might adapt to low light conditions by altering their growth patterns or metabolic processes to optimize energy use. ')
                print()
            elif result == 9:
                print('Temperature, Light Conditions are High and Soil Condition is Low.\n This can negatively impact root development, nutrient availability, and overall plant growth.')
                print()
            elif result == 10: 
                print('Light, Soil Conditions are Low and Temperature Condition is High.\nHigh temperatures influence plant growth rates, soil microbial activity, and overall metabolic functions. ')
                print()
            elif result == 11: 
                print('Light, Soil Conditions are High and Temperature Condition is Low.\n  This supports healthy root development, nutrient availability, and vigorous plant growth.')
                print()
            elif result == 12:
                print('Temperature, Light Conditions are Low and Soil Condition is High.\n Plants might show signs of stunted growth or reduced metabolic activity.')
                print()
            elif result == 13: 
                print('Temperature, Soil Conditions are Low and Light Condition is High.\n This could potentially slow down overall plant development.')
                print()
            elif result == 14: 
                print('Temperature, Soil, Light Conditions are Low.\n  Plants might exhibit stunted growth, reduced metabolic activity, or struggle to access essential nutrients for their development.')
                print()
            elif result == 15:
                print('Temperature, Soil Conditions are Normal and Light Condition is Low.\n  These challenges by hindering photosynthesis and overall plant development')
                print()
            elif result == 16: 
                print('Temperature, Light Conditions are Normal and Soil Condition is Low.\nIt poses challenges for optimal plant growth and productivity ')
                print()
            elif result == 17: 
                print('Light, Soil Conditions are Low and Temperature Condition is Normal.\n Plants might exhibit slower growth, reduced flowering, or delayed maturity due to limited energy production.')
                print()
            elif result == 18:
                print('Light, Soil Conditions are Normal and Temperature Condition is Low.\n This could potentially slow down overall plant development.')
                print()
            elif result == 19: 
                print('Temperature, Light Conditions are Low and Soil Condition is Normal.\nThe amount of sunlight reaching plants, affecting photosynthesis and potentially slowing down plant growth and productivity. ')
                print()
            elif result == 20: 
                print('Temperature, Soil Conditions are Low and Light Condition is Normal.\n It can affect plant growth rates, soil microbial activity, and metabolic functions. ')
                print()
            elif result == 21:
                print('Temperature Condition is Normal, Soil Condition is High and Light  Condition is Low \n it provides the necessary nutrients and support for root growth and development.')
                print()
            elif result == 22: 
                print('Temperature Condition is Normal, Soil Condition is Low and Light  Condition is High \nThey facilitate balanced growth and metabolic activities necessary for plant development. ')
                print()
            elif result == 23: 
                print('Temperature Condition is High, Soil Condition is Normal and Light  Condition is Low \n Plants might face stress due to the combination of high temperatures and inadequate light levels')
                print()
            elif result == 24:
                print('Temperature Condition is Low, Soil Condition is Normal and Light  Condition is High \n This could potentially slow down overall plant development.')
                print()
            elif result == 25: 
                print('Temperature Condition is Low, Soil Condition is High and Light  Condition is Normal \n This can ensure a more stable and consistent water supply for plants.')
                print()
            elif result == 26: 
                print('Temperature Condition is High, Soil Condition is Low and Light  Condition is Normal \n Plants may struggle to access sufficient water, making irrigation more critical. ')
                print()
    
            new_data = []
       else:
            print('Please check the sensor')
            print('_' * 80)
except ValueError:
    print("Invalid input or you chose to exit. Exiting the loop.")
