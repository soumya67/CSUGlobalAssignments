"""
Module 2: Critical Thinking
Option #1: KNN Classifier with Iris Data

Example of run:
python knn_iris.py 5.1 3.5 1.4 0.2

Results:
The predicted iris type is: Iris-setosa

Refereces
Microsoft & OpenAI. (2024). Bing Chat [GPT-4 language model]. 

"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sys

# Load the iris dataset from the local file path
data = pd.read_csv('E:\\MaintenanceScripts\\iris.csv')

# Split the dataset into features and target variable
X = data.iloc[:, :-1]  # Features: sepal length, sepal width, petal length, petal width
y = data.iloc[:, -1]   # Target: species

# Initialize the KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the data
knn.fit(X, y)

# Function to predict the iris type based on input features
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = knn.predict(input_data)
    return prediction[0]

# Main function to accept command line arguments and predict iris type
def main():
    if len(sys.argv) != 5:
        print("Usage: python knn_iris.py <sepal_length> <sepal_width> <petal_length> <petal_width>")
        return

    sepal_length = float(sys.argv[1])
    sepal_width = float(sys.argv[2])
    petal_length = float(sys.argv[3])
    petal_width = float(sys.argv[4])

    result = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    print(f"The predicted iris type is: {result}")

if __name__ == "__main__":
    main()
