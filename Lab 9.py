x = [1,2,3,4,5,6]
y = [2,4,6,8,10,12]

w = 0

lr = 0.1

batch_size = 2

epochs = 20

for epoch in range(epochs):

    for start in range(0, len(x), batch_size):

        xb = x[start:start+batch_size]
        yb = y[start:start+batch_size]

        dw = 0

        for i in range(len(xb)):
            prediction = w * xb[i]
            dw += (-2/len(xb))*xb[i]*(yb[i]-prediction)

        w = w - lr*dw

print("Final Weight =", round(w,4))
