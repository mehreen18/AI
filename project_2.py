import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(page_title="Salary Predictor", layout="wide")
st.title("💼 Salary Prediction — Neural Network")


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


# ---- 1. File upload ----
uploaded_file = st.file_uploader("Apload your CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna()

    st.subheader("Data Preview")
    st.dataframe(df.head(100))
    st.write(f"Shape: {df.shape}")

    target_column = "salary"
    if target_column not in df.columns:
        st.error(f"'{target_column}' column is not in the csv.file. Columns: {df.columns.tolist()}")
        st.stop()

    # Columns jo unique identifiers hain (Name, Email, Phone Number waghera) —
    # inka salary se koi meaningful relation nahi hota, isliye model se bahar rakhein.
    id_like_columns = st.multiselect(
        "Columns need to be ignored (like Name, Email, Number)?",
        options=[c for c in df.columns if c != target_column],
        default=[c for c in ["Name", "Email", "Number"] if c in df.columns],
    )

    X = df.drop(columns=[target_column] + id_like_columns)
    Y = df[target_column].to_numpy(dtype=np.float64)

    # ---- Encode categorical columns, keep encoders for later use ----
    encoders = {}
    for col in X.select_dtypes(include=['object', 'str']):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)

    Y_scaler = StandardScaler()
    Y_scaled = Y_scaler.fit_transform(Y.reshape(-1, 1)).flatten()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y_scaled, test_size=0.2, random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

    # ---- 2. Sidebar controls ----
    st.sidebar.header("Training Settings")
    epochs = st.sidebar.slider("Epochs", 100, 5000, 2000, step=100)
    lr = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01], value=0.001)

    if st.button("🚀 Train Model"):
        model = NeuralNetwork(input_size=X_train_t.shape[1])
        fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        losses = []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t).squeeze()
            loss = fn(outputs, Y_train_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                losses.append(loss.item())
                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"Epoch [{epoch+1}/{epochs}] — Loss: {loss.item():.4f}")
                chart_placeholder.line_chart(losses)

        st.success("Training complete ✅")

        # ---- 3. Evaluation on test set ----
        model.eval()
        with torch.no_grad():
            predictions_scaled = model(X_test_t).squeeze()
            predictions_original = Y_scaler.inverse_transform(
                predictions_scaled.numpy().reshape(-1, 1)
            )
            actual_original = Y_scaler.inverse_transform(
                Y_test_t.numpy().reshape(-1, 1)
            )

        st.subheader("Predictions vs Actual (sample)")
        results_df = pd.DataFrame({
            "Predicted": predictions_original.flatten()[:15],
            "Actual": actual_original.flatten()[:15],
        })
        results_df["Difference"] = (results_df["Predicted"] - results_df["Actual"]).abs()
        st.dataframe(results_df.style.format("{:,.0f}"))

        # ---- 4. Save everything needed for new predictions later ----
        st.session_state["model"] = model
        st.session_state["scalar"] = scalar
        st.session_state["Y_scaler"] = Y_scaler
        st.session_state["encoders"] = encoders
        st.session_state["feature_columns"] = X.columns.tolist()
        st.session_state["trained"] = True

        model_path = "employes_model.pth"
        torch.save(model.state_dict(), model_path)
        with open(model_path, "rb") as f:
            st.download_button(
                "⬇️ Download Trained Model (.pth)",
                data=f,
                file_name="employes_model.pth",
            )

    # ---- 5. Predict salary for a NEW employee ----
    if st.session_state.get("trained"):
        st.divider()
        st.subheader("🔮 Naye Employee ki Salary Predict Karein")

        feature_columns = st.session_state["feature_columns"]
        encoders = st.session_state["encoders"]

        with st.form("predict_form"):
            new_data = {}
            for col in feature_columns:
                if col in encoders:
                    # categorical column -> dropdown of known categories
                    options = list(encoders[col].classes_)
                    new_data[col] = st.selectbox(col, options)
                else:
                    # numeric column -> number input
                    default_val = float(df[col].mean())
                    new_data[col] = st.number_input(col, value=default_val)

            submitted = st.form_submit_button("Predict Salary")

        if submitted:
            model = st.session_state["model"]
            scalar = st.session_state["scalar"]
            Y_scaler = st.session_state["Y_scaler"]

            # Build a single-row dataframe in the same column order as training
            input_df = pd.DataFrame([new_data])[feature_columns]

            # Apply same label encoders used during training
            for col, le in encoders.items():
                input_df[col] = le.transform(input_df[col])

            # Apply same scaler used during training
            input_scaled = scalar.transform(input_df)
            input_t = torch.tensor(input_scaled, dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                pred_scaled = model(input_t).squeeze()
                pred_salary = Y_scaler.inverse_transform(
                    pred_scaled.numpy().reshape(-1, 1)
                )

            st.success(f"### Predicted Salary: {pred_salary.flatten()[0]:,.0f}")

else:
    st.info("Apload your CSV file before start.")