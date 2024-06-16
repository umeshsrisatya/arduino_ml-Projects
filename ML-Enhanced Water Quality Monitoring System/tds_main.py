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

dataset=pd.read_csv(r"C:\placement\aditiya__college\aditiya college\tds_sensor\tds_data.csv")

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

model=LogisticRegression()
model.fit(x_train,y_train)

x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy score for LOGR train data',training_data_accuracy*100)

x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy score for LOGR test data',training_data_accuracy*100)


# Visualize target variable distribution (Histogram)
plt.figure(figsize=(8, 6))
sns.histplot(y)
plt.title('Target Variable Distribution')
plt.xlabel('Result')
plt.ylabel('Count')
plt.show()

# Define Classifier names and their corresponding test accuracies
classifier_names=['RFC','DTC','KNN','SVM','LR']
train_accuracies=[100.0,100.0,93.9,100.0,97.4]
test_accuracies=[100.0,100.0,93.9,100.0,97.7]

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
    new_data = []
    while True:
        tds_data = ser.readline().decode('utf-8').strip()
        print("Tds sensor data:", tds_data)
        tds_data = float(tds_data)
        new_data.append([tds_data])  # Convert to float

        if len(new_data) >= 0:  
            
            predictions = model.predict(np.array(new_data))
            result = predictions[-1]  # Check the correct index for predictions
            #print(result)
            if result == 0:
                print('This TDS Range indicates water is Good for drinking.\n While pure water is essential for hydration, a lack of minerals over the long term could potentially lead to mineral deficiencies.')
                print()
            elif result == 1: 
                print('This water level indicates  water is Execellent for Drinking.\n Minerals are balanced condition in the body')
                print()
            elif result == 2:
                print('This water level indicates  water is Poor for Drinking.\n These contaminants could include heavy metals, pesticides, industrial waste. \n Long-term exposure to such contaminants can pose severe health risks including organ damage and an increased risk of cancer.')
                print()
            elif result == 3: 
                print('This water level indicates  water is worst for Drinking.\n Prolonged consumption of water with very high TDS might lead to mineral imbalances in the body.')
                print()
            new_data = []
        else:
            print('Please check the sensor')
            print('_' * 80)
except ValueError:
    print("Invalid input or you chose to exit. Exiting the loop.")
