# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: K.SUNDARESH
RegisterNumber:  212225220111
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("50_Startups.csv")
x = data["R&D Spend"].values
y = data["Profit"].values

x_mean = np.mean(x)

x_std = np.std(x)

x = (x - x_mean) / x_std

w = 0.0
b = 0.0
alpha = 0.01

epochs = 100

n = len(x)
losses = []

for _ in range(epochs):
    y_hat = w * x + b
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)
    
    dw = (2/n) * np.sum((y_hat - y)**2)
    losses.append(loss)
    
    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)
    
    w -= alpha * dw
    b -= alpha * db

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.xlabel("No of Iterations")
plt.ylabel("Loss")
plt.title("LOSS VS ITERATIONS")

plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 2)
plt.scatter(x, y, color="red", label="Data")
plt.plot(x, w * x + b, color="green", label="Regression Line")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("PROFIT VS R&D SPEND")
plt.legend()

plt.tight_layout()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)
```

## Output:
<img width="1015" height="635" alt="Screenshot 2026-01-31 090618" src="https://github.com/user-attachments/assets/60442702-08e6-4f6f-96fb-30951b6f25cd" />

<img width="912" height="639" alt="Screenshot 2026-01-31 090643" src="https://github.com/user-attachments/assets/cef66e70-0865-44a8-932f-da5985894a37" />


```
Final weight (w): 33671.51979690389
Final bias (b): 97157.57273469678

```

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
