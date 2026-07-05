#  neural network
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