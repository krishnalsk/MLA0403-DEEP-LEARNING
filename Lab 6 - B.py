from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron


# Load dataset

iris = load_iris()

X = iris.data
Y = iris.target


# Split data

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



# Single Layer Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

p_pred = perceptron.predict(X_test)

p_accuracy = accuracy_score(Y_test, p_pred)



# Multilayer Perceptron

mlp = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='relu',
    solver='adam',
    max_iter=3000,
    random_state=1
)


mlp.fit(X_train, Y_train)

m_pred = mlp.predict(X_test)

mlp_accuracy = accuracy_score(Y_test, m_pred)



print("Perceptron Accuracy:",
      round(p_accuracy*100,2), "%")

print("MLP Accuracy:",
      round(mlp_accuracy*100,2), "%")
