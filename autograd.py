import torch
x=torch.tensor(5.0, requires_grad=True)
##y=x**2
a=x+2
y=x**a
y.backward()
print(x.grad)



##  neural network
import torch
import torch.nn as nn

class MyNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(4, 16), 
            nn.ReLU(),

            nn.Linear(16, 32), 
            nn.ReLU(),

            nn.Linear(32, 64), 
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),  
            nn.ReLU(),

            nn.Linear(16, 2)   
        )

    def forward(self, x):
        return self.network(x)


model = MyNeuralNetwork()

print(model)

x=torch.tensor([[1.23,2.34,3.45,5.0]])

output=model(x)
print(x)
print(output)


## autograd

import torch
x=torch.tensor(5.0, requires_grad=True)
##y=x**2
a=x+2
y=x**a
y.backward()
print(x.grad)



##  neural network
import torch
import torch.nn as nn

class MyNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(4, 16), 
            nn.ReLU(),

            nn.Linear(16, 32), 
            nn.ReLU(),

            nn.Linear(32, 64), 
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),  
            nn.ReLU(),

            nn.Linear(16, 2)   
        )

    def forward(self, x):
        return self.network(x)


model = MyNeuralNetwork()

print(model)

x=torch.tensor([[1.23,2.34,3.45,5.0]])

output=model(x)
print(x)
print(output)



##Loss functions
import torch
import  torch.nn as nn
loss_fn=nn.MSELoss()

prediction=torch.tensor([280.0])
actual=torch.tensor([300.0])
loss=loss_fn(prediction,actual)
print(loss)
## 2nd prediction

import torch
import torch.nn as nn
loss_fn=nn.MSELoss()
prediction=torch.tensor([289.0])
actual=torch.tensor([300.0])
loss=loss_fn(prediction,actual)
print(loss)



## AI genration
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Step 1: Load Dataset
# -----------------------------

df = pd.read_csv("salary.csv")

print(df.head())

# -----------------------------
# Step 2: Convert CSV to Tensor
# -----------------------------

x = torch.tensor(
    df["Experience"].values,
    dtype=torch.float32
).view(-1,1)

y = torch.tensor(
    df["Salary"].values,
    dtype=torch.float32
).view(-1,1)

# -----------------------------
# Step 3: Create Model
# -----------------------------

model = nn.Linear(1,1)

# -----------------------------
# Step 4: Loss Function
# -----------------------------

loss_fn = nn.MSELoss()

# -----------------------------
# Step 5: Optimizer
# -----------------------------

optimizer = optim.SGD(
    model.parameters(),
    lr=0.001
)

# -----------------------------
# Step 6: Training
# -----------------------------

epochs = 5000

for epoch in range(epochs):

    prediction = model(x)

    loss = loss_fn(prediction,y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch+1)%500==0:
        print(
            f"Epoch {epoch+1} Loss = {loss.item():.2f}"
        )

# -----------------------------
# Step 7: Check Learned Values
# -----------------------------

print("\nWeight:",model.weight.item())
print("Bias:",model.bias.item())

# -----------------------------
# Step 8: Prediction
# -----------------------------

experience = torch.tensor([[12.0]])

salary = model(experience)

print("\nPredicted Salary")

print(salary.item())



## epoch
dataset=[1,2,3,4,5]
epochs = 3

for epoch in range(epochs):

    print("Epoch:", epoch+1)

    for data in dataset:

        print("Training on", data)

## loop complete

import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# Device (GPU if available)
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

# ==========================================
# Dataset
# Formula:
# y = 3*x1 + 2*x2 - x3 + 4*x4 + 5
# ==========================================

x = torch.tensor([
    [1.,2.,3.,4.],
    [2.,1.,4.,3.],
    [3.,2.,5.,1.],
    [4.,3.,2.,5.],
    [5.,4.,1.,2.],
    [6.,5.,3.,4.],
    [7.,6.,2.,5.],
    [8.,7.,4.,3.]
], dtype=torch.float32).to(device)

y =torch.tensor ([
    [25.],
     [21.],
     [17.],
     [40.],
     [35.],
     [48.],
     [58.],
     [46.]  
], dtype=torch.float32).to(device)

# ==========================================
# Neural Network
# ==========================================

class MyNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(4,16),
            nn.ReLU(),

            nn.Linear(16,32),
            nn.ReLU(),

            nn.Linear(32,64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,1)

        )

    def forward(self,x):
        return self.network(x)

# ==========================================
# Create Model
# ==========================================

model = MyNeuralNetwork().to(device)

print(model)

# ==========================================
# Loss Function
# ==========================================

loss_fn = nn.MSELoss()

# ==========================================
# Optimizer
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

# ==========================================
# Training
# ==========================================

epochs = 2000

for epoch in range(epochs):

    prediction = model(x)

    loss = loss_fn(prediction,y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch+1)%100==0:

        print(
            f"Epoch {epoch+1:4d} | Loss = {loss.item():.6f}"
        )

# ==========================================
# Learned Parameters
# ==========================================

print("\nModel Parameters\n")

for name,param in model.named_parameters():

    print(name)
    print(param.shape)
    print()

# ==========================================
# Testing
# ==========================================

print("\nTesting\n")

test = torch.tensor([[10.,8.,3.,7.]],dtype=torch.float32).to(device)

prediction = model(test)

print("Prediction:",prediction.item())


