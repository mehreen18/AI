import torch
import os
import streamlit as st
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv(r"C:\Users\Mehreen\Documents\GitHub\AI\__pycache__\employes.csv")
df = df.dropna()
print(df)
print(df.shape)
print(df.columns.tolist())

target_column = "salary"
X = df.drop(columns=[target_column])
Y = df[target_column].to_numpy(dtype=np.float64)

for col in X.select_dtypes(include=['object', 'str']):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

Y_scaler = StandardScaler()
Y_scaled = Y_scaler.fit_transform(Y.reshape(-1, 1)).flatten()

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y_scaled,        # <-- fixed: Y_scaled, not Y
    test_size=0.2,
    random_state=42,
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.r1(self.fc1(x))
        x = self.r2(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNetwork(input_size=X_train_t.shape[1])

fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t).squeeze()
    loss = fn(outputs, Y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    predictions_scaled = model(X_test_t).squeeze()
    predictions_original = Y_scaler.inverse_transform(predictions_scaled.numpy().reshape(-1, 1))
    actual_original = Y_scaler.inverse_transform(Y_test_t.numpy().reshape(-1, 1))

    print("\n--- Sample Predictions vs Actual ---")
    for pred, actual in list(zip(predictions_original.flatten(), actual_original.flatten()))[:15]:
        diff = abs(pred - actual)
        print(f"Predicted: {pred:,.0f}   |   Actual: {actual:,.0f}   |   Difference: {diff:,.0f}")

torch.save(model.state_dict(),r"C:\Users\Mehreen\Documents\GitHub\AI\__pycache__\employes.csv")
print("model saved")



####  relaoad model again
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================
# STEP 1: Wahi preprocessing setup jo training mein tha
# (Zaroori hai kyunke naye data ko bhi usi tarah scale karna hai)
# ============================================
df = pd.read_csv(r"C:\Users\Mehreen\Documents\GitHub\AI\__pycache__\employes.csv")
df = df.dropna()

target_column = "salary"
X = df.drop(columns=[target_column])
Y = df[target_column].to_numpy(dtype=np.float64)

for col in X.select_dtypes(include=['object', 'str']):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

scalar = StandardScaler()
scalar.fit(X)          # sirf fit karna hai, transform training data pe nahi karna

Y_scaler = StandardScaler()
Y_scaler.fit(Y.reshape(-1, 1))

# ============================================
# STEP 2: Wahi model architecture dobara define karein
# (PyTorch ko pata hona chahiye model ka structure kya hai)
# ============================================
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.r1(self.fc1(x))
        x = self.r2(self.fc2(x))
        x = self.fc3(x)
        return x


model = NeuralNetwork(input_size=X.shape[1])
model.load_state_dict(torch.load(r"C:\Users\Mehreen\Documents\GitHub\AI\salary_model.pth"))
model.eval()
print("Model loaded successfully!")


new_person = pd.DataFrame({
    'Name': ['Ahmed'],
    'Age': [35],
    'Number': [3123456789],
    'Email': ['ahmed@gmail.com'],
    'duration': [60],
    'year': [2022],
    'month': [5]
})

for col in new_person.select_dtypes(include=['object', 'str']):
    le = LabelEncoder()
    new_person[col] = le.fit_transform(new_person[col])

new_scaled = scalar.transform(new_person)
new_tensor = torch.tensor(new_scaled, dtype=torch.float32)

with torch.no_grad():
    pred_scaled = model(new_tensor).squeeze()
    pred_salary = Y_scaler.inverse_transform(pred_scaled.numpy().reshape(-1, 1))
    print(f"Predicted Salary: {pred_salary[0][0]:,.0f}")