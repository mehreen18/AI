import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )
print("available device is :" ,device)

x=torch.tensor(
    [
        [1.,2.,3.,4.],
        [4.,3.,2.,1.],
        [9.,8.,7.,6.],
        [3.,5.,5.,7.],
        [2.,3.,4.,5.],
        [10.,8.,5.,7.],
        [10.,2.,8.,9.],
        [10.,10.,10.,10.],
        [20.,20.,20.,20.]

    ], dtype=torch.float32
).to(device)

y=torch.tensor([
    [48.],
    [34.],
    [109.],
    [92.],
    [66.],
    [119.],
    [128.],
    [145.],
    [305.]

], dtype=torch.float32
   
).to(device)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.Network=nn.Sequential(
            nn.Linear(4,16),
            nn.ReLU(),

            nn.Linear(16,32),
            nn.ReLU(),

            nn.Linear(32,64),
            nn.ReLU(),

            nn.Linear(64,16),
            nn.ReLU(),

            nn.Linear(16,1)

        )
    def forward(self,x):
        return self.Network(x)
    

model=NeuralNetwork().to(device)
print(model)    


loss_fn=nn.MSELoss()

optimizer=optim.Adam(
model.parameters(),
lr=0.01
)

epochs=3000
for epoch in range(epochs):
    prediction=model(x)

    loss=loss_fn(prediction,y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if(epoch+1)%100==0:
        print(f"epoch {epoch+1 :4d} loss= {loss.item():.6f}")

print("-----MODEL PARAMETERS-----")
for name,param in model.named_parameters():
    print(name)
    print(param.shape)
    print()

 ## test

test=torch.tensor([[20.,20.,20.,20.]] , dtype=torch.float32).to(device)
prediction=model(test)
print(" prediction : ", prediction.item())
              



