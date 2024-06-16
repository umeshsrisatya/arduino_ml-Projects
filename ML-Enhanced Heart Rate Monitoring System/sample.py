import serial
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

warnings.filterwarnings('ignore')

# Function to calculate stress percentage
def calculate_stress_percentage(current_heart_rate, resting_heart_rate, heart_rate_reserve):
    return ((current_heart_rate - resting_heart_rate) / heart_rate_reserve) * 100

# Read dataset
dataset = pd.read_csv(r"c:\Users\KUMAR\Desktop\Yashwanth\Arduino_hb\heart_beat\hb_data.csv")

# Print dataset information
print(dataset.head())
print(dataset.shape)
print(dataset.describe())
print(dataset.isna().sum())

# Separate features and target variable
x = dataset.drop('result', axis=1)
y = dataset['result']

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Train Random Forest Classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Print training accuracy
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy score for RFC train data:', training_data_accuracy * 100)

# Print testing accuracy
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy score for RFC test data:', test_data_accuracy * 100)

# Define Classifier names and their corresponding test accuracies
classifier_names = ['RFC']
train_accuracies = [training_data_accuracy * 100]
test_accuracies = [test_data_accuracy * 100]

# Plot training and testing accuracies
plt.figure(figsize=(10, 6))
width = 0.35
x = range(len(classifier_names))
plt.bar(x, train_accuracies, width, label='Training Accuracy', alpha=0.7)
plt.bar([i + width for i in x], test_accuracies, width, label='Testing Accuracy', alpha=0.7)
plt.xlabel('Classifiers')
plt.ylabel('Accuracy(%)')
plt.title('Classifier Training and Testing Accuracy Comparision')
plt.xticks([i + width / 2 for i in x], classifier_names)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.show()

# Box Plot graph
plt.figure(figsize=(10, 6))
sns.boxplot(data=dataset.drop('result', axis=1))
plt.title('Boxplot of Features')
plt.xticks(rotation=45)
plt.show()

try:
    ser = serial.Serial('COM6', 9600)
    new_data = []  # Initialize a list to store sensor data
    
    # Read data from Arduino
    heart_rate = ser.readline().decode('utf-8').strip()
    print("Heart Beat sensor data:", heart_rate)
    heart_rate = int(heart_rate)
    new_data.append([heart_rate])

    if len(new_data) >= 0:
        # Predict stress level using the trained model
        predictions = model.predict(np.array(new_data))
        result = predictions[-1]  # Check the correct index for predictions

        # Define stress level ranges and corresponding messages
        stress_levels = {
            0: "This level indicates a heart rate significantly below the normal range, known as bradycardia",
            1: "A heart rate within the typical, healthy range for an individual at rest",
            2: "Moderate heart rate detected. This level indicates a heart rate that may be influenced by various factors such as stress, physical exertion, or certain medical conditions.",
            3: "High heart rate detected. This level indicates a heart rate that exceeds the normal range, often referred to as tachycardia. It may indicate stress, physical exertion, or certain medical conditions."
        }

        # Get the stress level message based on the detected result
        stress_message = stress_levels.get(result, "Invalid stress level detected.")

        # Print the stress level message
        print(stress_message)

        # Reset new_data after processing
        new_data = []
    else:
        print('Please check the sensor')
        print('_' * 80)
except ValueError:
    print("Invalid input or you chose to exit. Exiting the loop.")
