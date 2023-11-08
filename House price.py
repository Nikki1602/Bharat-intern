# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the example dataset
data = """SquareFeet, Bedrooms, Bathrooms, YearBuilt, Price
1500, 3, 2, 1990, 200000
2000, 4, 2.5, 2005, 280000
1200, 2, 1.5, 1985, 150000
1800, 3, 2, 1998, 230000
2500, 4, 3, 2010, 350000"""
dataset = pd.read_csv(pd.compat.StringIO(data))

# Assume 'SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt' are features, and 'Price' is the target
X = dataset[['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt']]
y = dataset['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
