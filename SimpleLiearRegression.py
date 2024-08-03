"""
Module 3: Critical Thinking
Option #1: Simple Linear Regression in Scikit Learn


Refereces
Microsoft & OpenAI. (2024). Bing Chat [GPT-4 language model]. 

"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
lr = LinearRegression()

# Fit the model on the training data
lr.fit(X_train, y_train)

# Predict the house values on the testing set
y_pred = lr.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print(f"The mean squared error of the linear regression model is {mse:.2f}")

# Demonstrate predicting the value of a house
# We will use the first row of the test set as an example
example_features = X_test[0]
predicted_value = lr.predict([example_features])[0]
print(f"Predicted value of the house with features {example_features} is {predicted_value:.2f}")
