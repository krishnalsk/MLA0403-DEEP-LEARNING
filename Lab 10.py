import numpy as np

dimensions = [2,5,10,20,50,100]

for d in dimensions:

    point1 = np.random.rand(d)
    point2 = np.random.rand(d)

    distance = np.linalg.norm(point1-point2)

    print("Dimension:", d,
          " Distance:", round(distance,4))
