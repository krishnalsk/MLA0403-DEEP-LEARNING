# Import libraries

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# Load dataset
iris = load_iris()

X = iris.data
Y = iris.target


# Split dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# Data preprocessing (Feature Scaling)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create Machine Learning Model

model = LogisticRegression()


# Train model

model.fit(X_train, Y_train)


# Prediction

Y_pred = model.predict(X_test)


# Performance Evaluation

accuracy = accuracy_score(Y_test, Y_pred)

cm = confusion_matrix(Y_test, Y_pred)


print("Accuracy:", round(accuracy*100,2), "%")

print("\nConfusion Matrix:")
print(cm)
