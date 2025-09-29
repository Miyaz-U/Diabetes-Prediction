# Importing the necessary libraries
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import h2o
import joblib
from dotenv import load_dotenv

# Initialize environment and database connection
load_dotenv()
db_url = os.getenv("db_url")
eng = ce(db_url)

# Initialize H2O
h2o.init()

# Load Data
data = pd.read_sql("SELECT * FROM diabetes_data;", eng)
data.dropna(inplace=True)

# Label encoding for categorical columns
encoder_gender = LabelEncoder()
encoder_smoking_history = LabelEncoder()
if "gender" in data.columns:
    data["gender"] = encoder_gender.fit_transform(data["gender"].astype(str))
if "smoking_history" in data.columns:
    data["smoking_history"] = encoder_smoking_history.fit_transform(data["smoking_history"].astype(str))

X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# Handling rare classes
counts = y.value_counts()
valid_classes = counts[counts >= 2].index
X = X[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]

# Load Best Model
with open("Models/BestModelAutomated.txt", "r") as f:
    best_model_name = f.read().strip()

is_h2o_model = False
if best_model_name in ["RandomForestClassifier", "RandomForestRegressor",
                       "LogisticRegression", "LinearRegression",
                       "XGBClassifier", "XGBRegressor"]:
    # sklearn model
    model = joblib.load("Models/BestModelAutomated.pkl")
else:
    # H2O model
    model = h2o.load_model("Models/BestModelAutomated")
    is_h2o_model = True

# Streamlit UI
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")

st.sidebar.image("https://img.freepik.com/free-vector/doctor-character-background_1270-84.jpg?w=2000", width="stretch")
st.sidebar.title("🏥 Diabetes Prediction")
st.sidebar.header("Enter Patient Details :- ")
st.sidebar.write(f"This app predicts the **Diabetes Level** using **{best_model_name}** model.")

# Model Performance
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, stratify=y)

# Predict
if is_h2o_model:
    x_test_h2o = h2o.H2OFrame(x_test)
    y_pred = model.predict(x_test_h2o).as_data_frame()["predict"]
else:
    y_pred = model.predict(x_test)

# Determine task type
is_classification = y.nunique() <= 2

# Threshold if classification
if is_classification and not is_h2o_model:
    y_pred = (y_pred > 0.5).astype(int)
if is_classification and is_h2o_model:
    y_pred = (y_pred > 0.5).astype(int)

st.sidebar.write("📊 Model Performance Metrics")
if is_classification:
    st.sidebar.write(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    st.sidebar.write(f"Precision Score: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    st.sidebar.write(f"Recall Score: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    st.sidebar.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
else:
    st.sidebar.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
    st.sidebar.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
    st.sidebar.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# Input Form
st.title(f"🏥 Diabetes Prediction App ({best_model_name})")
st.header("👩‍⚕️ Enter Patient Details:")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("🧑‍🦱 Gender", ["Male", "Female", "Transgender"])
    hypertension = st.selectbox("⚡ Hypertension", ["Select", "Yes", "No"])
    smoking_history = st.selectbox("🚬 Smoking History", ["never", "current", "former"])
    know_bmi = st.toggle("👨‍⚕️ Do you know your BMI?")
with col2:
    age = st.number_input("👶 Age", min_value=1, max_value=120)
    heart_disease = st.selectbox("🫀 Heart Disease", ["Select", "Yes", "No"])
    hba1c_level = st.number_input("🩸 HbA1c Level (as %)", min_value=1.0, max_value=20.0)
    bmi = None
    if know_bmi:
        bmi = st.number_input("⚖️🧍📏 BMI", min_value=10.0, max_value=70.0)
    else:
        weight = st.number_input("⚖️ Weight (kg)", min_value=1.0, max_value=300.0)
        height = st.number_input("📏 Height (cm)", min_value=30.0, max_value=250.0)
        bmi = round(weight / ((height / 100) ** 2), 2)
        st.info(f"⚖️🧍📏 Calculated BMI: {bmi}")

# Encode Inputs
gender_num = encoder_gender.transform([gender])[0]
smoking_history_num = encoder_smoking_history.transform([smoking_history])[0]
hypertension_num = 1 if hypertension == "Yes" else 0
heart_disease_num = 1 if heart_disease == "Yes" else 0

user_df = pd.DataFrame([{
    "gender": gender_num,
    "age": age,
    "hypertension": hypertension_num,
    "heart_disease": heart_disease_num,
    "smoking_history": smoking_history_num,
    "bmi": bmi,
    "hba1c_level": hba1c_level
}])

# Prediction
if st.button("Predict Diabetes Level"):
    if hypertension == "Select" or heart_disease == "Select":
        st.error("⚠️ Please enter all the values.")
    else:
        if is_h2o_model:
            user_h2o = h2o.H2OFrame(user_df)
            prediction = model.predict(user_h2o).as_data_frame().iloc[0, 0]
        else:
            prediction = model.predict(user_df)[0]

        st.subheader("📊 Prediction Result")
        st.write(f"🧪 Predicted diabetes level: {prediction}")

        # Threshold if classification
        if is_classification:
            if prediction >= 0.5:
                st.error("🚨 The patient is likely to have diabetes.")
                predicted_class = 1
            else:
                st.success("✅ The patient is unlikely to have diabetes.")
                predicted_class = 0
        else:
            predicted_class = prediction

        # Save to DB
        save_df = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "hypertension": hypertension_num,
            "heart_disease": heart_disease_num,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "hba1c_level": hba1c_level,
            "diabetes": predicted_class
        }])
        try:
            save_df.to_sql("diabetes_data", eng, if_exists="append", index=False)
            st.success("📝 User inputs and predictions stored successfully.")
        except Exception as e:
            st.error(f"Error while saving prediction: {e}")