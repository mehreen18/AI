import torch
import os
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
print(os.getcwd())               # abhi Python kis folder mein hai
print(os.path.exists("employes.csv"))  
print(os.listdir()) 
# df = pd.read_csv("employes.csv")
# df = df.dropna()
# print(df)
# print(df.shape)
# print(df.columns.tolist())

# target_column = "salary"
# X = df.drop(columns=[target_column])
# Y = df[target_column].to_numpy(dtype=np.float64)

# for col in X.select_dtypes(include=['object', 'str']):
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col])

# scalar = StandardScaler()
# X_scaled = scalar.fit_transform(X)

# Y_scaler = StandardScaler()
# Y_scaled = Y_scaler.fit_transform(Y.reshape(-1, 1)).flatten()

# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_scaled, Y_scaled,        # <-- fixed: Y_scaled, not Y
#     test_size=0.2,
#     random_state=42,
# )

# X_train_t = torch.tensor(X_train, dtype=torch.float32)
# X_test_t = torch.tensor(X_test, dtype=torch.float32)
# Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
# Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.r1 = nn.ReLU()
#         self.fc2 = nn.Linear(64, 32)
#         self.r2 = nn.ReLU()
#         self.fc3 = nn.Linear(32, 1)

#     def forward(self, x):
#         x = self.r1(self.fc1(x))
#         x = self.r2(self.fc2(x))
#         x = self.fc3(x)
#         return x

# model = NeuralNetwork(input_size=X_train_t.shape[1])

# fn = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# epochs = 1000
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train_t).squeeze()
#     loss = fn(outputs, Y_train_t)
#     loss.backward()
#     optimizer.step()

#     if (epoch+1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# model.eval()
# with torch.no_grad():
#     predictions = model(X_test_t).squeeze()
#     test_loss = fn(predictions, Y_test_t)
#     print(f"Test Loss: {test_loss.item():.4f}")
