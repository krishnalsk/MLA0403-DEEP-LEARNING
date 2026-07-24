# Gradient Descent for y = 2x

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

w = 0
learning_rate = 0.1
epochs = 100

n = len(x)

for epoch in range(epochs):

    dw = 0

    for i in range(n):
        prediction = w * x[i]
        dw += (-2 / n) * x[i] * (y[i] - prediction)

    w = w - learning_rate * dw

print("Final Weight =", round(w,4))
