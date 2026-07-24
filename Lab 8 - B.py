x = [1,2,3,4,5]
y = [2,4,6,8,10]

w = 0
lr = 0.1

epochs = 20

for epoch in range(epochs):

    for i in range(len(x)):

        prediction = w * x[i]

        error = y[i] - prediction

        gradient = -2 * x[i] * error

        w = w - lr * gradient

print("Final Weight =", round(w,4))
