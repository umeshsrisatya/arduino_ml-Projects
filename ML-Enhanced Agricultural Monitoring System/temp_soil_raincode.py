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

dataset=pd.read_csv(r"C:\placement\aditiya__college\aditiya college\temp_soil_rain\rain_soil_temp_data.csv")

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
train_accuracies=[100.0,100.0,93.9,95.0]
test_accuracies=[100.0,100.0,93.9,98.0]

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

#Violinplot to visualize feature distributions:
plt.figure(figsize=(10, 6))
sns.violinplot(data=dataset.drop('result', axis=1), scale='width')
plt.title('Violinplot of Features')
plt.xticks(rotation=45)
plt.show()

try:
    ser = serial.Serial('COM4', 9600)
    new_data = []  # Initialize a list to store sensor data
    while True:
       data = ser.readline().decode().strip()
       sensor_values = data.split(',')
    # Extract sensor values
       rain_value = float(sensor_values[0])
       soil_value = float(sensor_values[1])
       temperature_value = float(sensor_values[2])

       print('rain_value', rain_value)
       print('soil_value', soil_value)
       print('temperature_value', temperature_value)

       new_data.append([rain_value,soil_value, temperature_value])

       if len(new_data) >= 0:  
            
            predictions = model.predict(np.array(new_data))
            result = predictions[-1]  # Check the correct index for predictions
            print(result)
            if result == 0:
                print('All conditions are Normal.')
                print()
            elif result == 1: 
                print('Rain, soil Conditions are Normal and Temperature Condition is High.\n The air might feel dry, and there is a reduced likelihood of immediate precipitation.')
                print()
            elif result == 2: 
                print('Rain, Temperature  Conditions are Normal and soil Condition is High \n  This combination provides adequate water and temperature levels for plants to thrive.')
                print()
            elif result == 3:
                print('Soil, Temperature  Conditions are High and Rain Condition is Normal\n This can lead to stress on vegetation, affecting agricultural crops, gardens, and natural vegetation.')
                print()
            elif result == 4: 
                print('Soil, Temperature  Conditions are Normal and Rain Condition is High \n Prolonged saturation might hinder plant growth, impact root health, and reduce oxygen availability in the soil.')
                print()
            elif result == 5: 
                print('Rain, Temperature  Conditions are High and soil Condition is Low \n Insufficient rainfall and high evaporation rates can exacerbate soil dryness, impacting agriculture, water resources, and ecosystems.')
                print()
            elif result == 6:
                print('Rain, Soil Conditions are Normal and Temperature Condition is High \n This combination results in higher overall moisture loss, potentially leading to drier surface conditions.')
                print()
            elif result == 7: 
                print('Rain, Soil ,Temperature Condition are High\n This combination results in higher overall moisture loss, potentially leading to drier surface conditions. ')
                print()
            elif result == 8: 
                print('Rain, soil Conditions are High and Temperature Condition is Low \n precautions should be taken to prevent soil erosion and ensure proper drainage while protecting against potential cold-related hazards.')
                print()
            elif result == 9:
                print('Rain, Temperature Conditions are High and soil Condition is Low \n Careful attention should be paid to soil hydration and irrigation to maintain soil health and support plant growth in hot weather conditions.')
                print()
            elif result == 10: 
                print('soil ,Temperature Conditions are Low and Rain Condition is High\n Potential risks include soil saturation, increased risk of flooding, and susceptibility to soil erosion, emphasizing the need for drainage management and soil conservation practices. ')
                print()
            elif result == 11: 
                print('Temperature, soil  Conditions are High and Rain Condition is Low\n Irrigation management is essential to ensure adequate hydration for plants and prevent soil dehydration, promoting healthy growth and vegetation.')
                print()
            elif result == 12:
                print('Rain, Temperature  Conditions are Low and soil Condition is High \n Emphasizing the need for proper drainage and soil management practices to prevent waterlogging and ensure healthy plant growth.')
                print()
            elif result == 13: 
                print('Rain, soil  Conditions are Low and Temperature Condition is High \n Importance of efficient irrigation and water conservation practices to maintain soil health and support plant growth in hot weather conditions. ')
                print()
            elif result == 14: 
                print('Rain,soil, Temperature  Conditions are Low \nThe need for water conservation measures, efficient irrigation practices, and soil moisture management to support plant growth and prevent soil dehydration. ')
                print()
            elif result == 15:
                print('Rain, soil  Conditions are Normal and Temperature Condition is Low \n To protect plants from cold stress and frost damage, such as covering sensitive plants or providing additional insulation, while also ensuring proper soil drainage to prevent waterlogging. ')
                print()
            elif result == 16: 
                print('Rain, Temperature  Conditions are Normal and soil Condition is Low \n Replenish soil moisture through irrigation or mulching to support plant growth and maintain soil health. ')
                print()
            elif result == 17: 
                print('soil, Temperature  Conditions are Low and Rain Condition is Normal \n Implementing measures such as mulching to retain moisture and support plant growth during cooler weather conditions. ')
                print()
            elif result == 18:
                print('Soil, Temperature  Conditions are Normal and Rain Condition is Low\n Low rainfall is crucial to monitor soil hydration carefully and implement appropriate irrigation practices to maintain soil health and support plant growth during periods of reduced precipitation.')
                print()
            elif result == 19: 
                print('Rain, Temperature  Conditions are Low and Soil Condition is Normal \n  its important to monitor soil moisture levels closely and consider supplemental watering to ensure adequate hydration for plants and maintain soil health during dry periods.')
                print()
            elif result == 20: 
                print('Rain, Soil  Conditions are Low and Temperature Condition is Normal \n With low moisture in the soil, there could also be concerns about water availability for plants and ecosystems.')
                print()
            elif result == 21:
                print('Rain Condition is Normal, Soil Condition is High and Temperature  Condition is Low \nHigher soil moisture levels could lead to increased microbial activity and nutrient availability in the soil.')
                print()
            elif result == 22: 
                print('Rain Condition is Normal, Soil Condition is Low and Temperature  Condition is High \n Plants may experience stress due to the combination of low soil moisture and high temperatures, potentially leading to wilt, reduced growth, or even plant death in extreme cases.')
                print()
            elif result == 23: 
                print('Rain Condition is High, Soil Condition is Normal and Temperature  Condition is Low \n Farmers may need to monitor their crops closely for signs of disease and take appropriate preventive measures.')
                print()
            elif result == 24:
                print('Rain Condition is Low, Soil Condition is Normal and Temperature  Condition is High\n Heat stress can lead to reduced photosynthesis, flower and fruit drop, and decreased overall productivity. ')
                print()
            elif result == 25: 
                print('Rain Condition is Low, Soil Condition is High and Temperature  Condition is Normal\n Supplemental irrigation may be necessary to maintain adequate soil moisture levels for optimal plant growth ')
                print()
            elif result == 26: 
                print('Rain Condition is High, Soil Condition is Low and Temperature  Condition is Normal \nThese practices can help enhance soil structure, fertility, and moisture retention over time. ')
    
            new_data = []
       else:
            print('Please check the sensor')
            print('_' * 80)
except ValueError:
    print("Invalid input or you chose to exit. Exiting the loop.")
