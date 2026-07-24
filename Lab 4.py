# Import required libraries

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Non-linear XOR dataset

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

Y = np.array([0,1,1,0])


# Split dataset

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, random_state=1
)


# Create Neural Network
# One hidden layer with 4 neurons

model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='relu',
    max_iter=1000,
    random_state=1
)


# Train model

model.fit(X_train, Y_train)


# Prediction

Y_pred = model.predict(X_test)


# Accuracy

accuracy = accuracy_score(Y_test, Y_pred)


print("Actual Output:", Y_test)
print("Predicted Output:", Y_pred)
print("Accuracy:", round(accuracy*100,2), "%")
