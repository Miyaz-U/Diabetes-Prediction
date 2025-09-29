# Importing the required modules and libraries
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine as ce
from sklearn.base import is_classifier, is_regressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score as r2
from sklearn.metrics import accuracy_score as acc, precision_score as ps, recall_score as rs, f1_score as f1
import joblib
import matplotlib.pyplot as plt

# Loading the dataset
# Database URL format: postgresql+psycopg2://<user>:<password>:<host>:<port>/<database>
#db_url = os.getenv("db_url")

# Load DB URL from secrets
db_url = st.secrets["database"]["url"]

# Connecting to PostgreSQL
eng = ce(db_url)

# Loading and viewing the dataset
data = pd.read_sql("select * from diabetes_data;", eng)

# Data Preprocessing

# Convert the categorical columns into numerical values
encoder_gender = LabelEncoder()
encoder_smoking_history = LabelEncoder()
data["gender"] = encoder_gender.fit_transform(data["gender"])
data["smoking_history"] = encoder_smoking_history.fit_transform(data["smoking_history"])

# Load the best model
model = joblib.load("Models/BestModelManual.pkl")

# Streamlit interface

st.markdown(
    """
    <style>
    /* Reducing the size of the input boxes */
    .stSelectbox > div > > div,
    .stNumberInput > div > div,
    .stTextInput > div > div,
    .stTextArea > div > div,
    .stToggle > div {
        font-size : 14px !important; /* Reduce font */
        height : 2.2 rem !important; /* Reduce Height */
        padding : 0.2 rem 0.5 rem !important;
    }

    /* Shrink Labels */
    label, .stMarkown {
        font-size : 14px !important;
    }

    /* Adjust Button Size */
    .stButton button {
        font-size : 14px !important;
        padding : 0.3 rem 0.8 rem !important;
    }
    </style>
    """,
    unsafe_allow_html = True
)

with open("Models/BestModelManual.txt", "r") as f:
    best_model_name = f.read().strip()

# Setting up the window
st.set_page_config(page_title = "Diabetes Predictor", page_icon="🩺", layout = "wide")
st.sidebar.image("https://img.freepik.com/free-vector/doctor-character-background_1270-84.jpg?w=2000", width = "stretch")  
st.sidebar.title("🏥Diabetes Prediction")
st.sidebar.header("Enter Patient Details :- ")
st.sidebar.write(f"""
                 This app predicts the **Diabetes Level** based on the patient details provided
                 using the **{best_model_name}** model.
                 """)

# Model Performance
x = data.drop("diabetes", axis = 1)
y = data["diabetes"]
# Handling rare classes : Keeping only the classes which has atleast two samples
counts = y.value_counts()
valid_classes = counts[counts >= 2].index
x = x[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =50, stratify = y)
y_prediction = model.predict(x_test)

# Displaying the model performance metrics in the sidebar
st.sidebar.write("📊Model Performance Metrics")
# Checking whether the model is a regressor
if is_regressor(model):
    # Model Performance Metrics (Regressor)
    mse_value = mse(y_test, y_prediction)
    mae_value = mae(y_test, y_prediction)
    r2_value = r2(y_test, y_prediction)
    # Displaying the model performance metrics on the sidebar
    st.sidebar.write(f"Mean Squared Error (MSE): {mse_value: .4f}.")
    st.sidebar.write(f"Mean Absolute Error (MAE): {mae_value: .4f}.")
    st.sidebar.write(f"R^2 Score: {r2_value: .4f}.")
elif is_classifier(model):
    # Model Performance Metrics (Classifier)
    accs = acc(y_test, y_prediction)
    prs = ps(y_test, y_prediction, average = "weighted", zero_division = 0)
    res = rs(y_test, y_prediction, average = "weighted", zero_division = 0)
    f1s = f1(y_test, y_prediction, average = "weighted", zero_division = 0)
    # Displaying the model performance metrics on the sidebar
    st.sidebar.write(f"Accuracy Score: {accs: .4f}.")
    st.sidebar.write(f"Precision Score: {prs: .4f}.")
    st.sidebar.write(f"Recall Score: {res: .4f}.")
    st.sidebar.write(f"F1 Score: {f1s: .4f}.")

st.title(f"🏥Diabetes Prediction App ({best_model_name})")

# Getting inputs from the user

st.header("👩‍⚕️Enter Patient Details:")

# Creating two columns for better display
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("🧑‍🦱Gender", ["Male", "Female", "Transgender"])
    hypertension = st.selectbox("⚡Hypertension", ["Select", "Yes", "No"])
    smoking_history = st.selectbox("🚬Smoking History", ["never", "current", "former"])
    know_bmi = st.toggle("👨‍⚕️Do you know your BMI?", ["Yes", "No"])
with col2:
    age = st.number_input("👶Age", min_value = 1, max_value = 120)
    heart_disease = st.selectbox("🫀Heart Disease", ["Select", "Yes", "No"])
    hba1c_level = st.number_input("🩸HbA1c Level (as a percentage)", min_value = 1.0, max_value = 20.0)
    bmi = None
    if know_bmi:
        bmi = st.number_input("⚖️🧍📏BMI", min_value = 10.0, max_value = 70.0)
    else:
       weight = st.number_input("⚖️Weight (in kg)", min_value = 1.0, max_value = 300.0)
       height = st.number_input("📏Height (in cm)", min_value = 30.0, max_value = 250.0)
       height_m = height /100
       bmi = round(weight / (height_m ** 2), 2)
       st.info(f"⚖️🧍📏Calculated BMI: {bmi}")

# Converting the categorical inputs into numerical inputs
gender_num = encoder_gender.transform([gender])[0]
smoking_history_num = encoder_smoking_history.transform([smoking_history])[0]
if hypertension == "Yes":
    hypertension_num = 1
else:
    hypertension_num = 0
if heart_disease == "Yes":
    heart_disease_num = 1
else:
    heart_disease_num = 0

# Creating a dataframe of user inputs for prediction
user_df = pd.DataFrame([{
    "gender" : gender_num,
    "age" : age,
    "hypertension" : hypertension_num,
    "heart_disease" : heart_disease_num,
    "smoking_history" : smoking_history_num,
    "bmi" : bmi,
    "hba1c_level" : hba1c_level
}])
    
# Prediction Button
if st.button("Predict Diabetes Level"):
    # Validation Check
    prediction = model.predict(user_df)[0]
    if hypertension == "Select" or heart_disease == "Select":
        st.error("⚠️ Please enter all the values.")
    else:
        st.subheader("📊Prediction Result")
        st.write(f"🧪The predicted diabetes level is: {prediction: .4f}.")
        # Final Prediction
        if prediction >= 0.5:
            st.error("🚨The patient is likely to have diabetes.")
            predicted_class = 1
        else:
            st.success("✅The patient is unlikely to have diabetes.")
            predicted_class = 0
        
        # Save user inputs and predictions to the dataset
        save_df = pd.DataFrame([{
            "gender" : gender,
            "age" : age,
            "hypertension" : 1 if hypertension == "Yes" else 0,
            "heart_disease" : 1 if heart_disease == "Yes" else 0,
            "smoking_history" : smoking_history,
            "bmi" : bmi,
            "hba1c_level" : hba1c_level,
            "diabetes" : predicted_class
        }])
        try:
            save_df.to_sql("diabetes_data", eng, if_exists = "append", index = False)
            print("The predictions has been saved to the database successfully.")
        except Exception as e:
            print(f"Error while storing prediction: {e}.")
        st.success("📝User inputs and predictions have been stored successfully.")
