# Importing the required modules and libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine as ce
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score as r2
# from matplotlib import pyplot as plt
from xgboost import XGBRegressor as xgbr
import joblib
from dotenv import load_dotenv
import os

load_dotenv()

# Database URL format: postgresql+psycopg2://<user>:<password>:<host>:<port>/<database>
db_url = os.getenv("db_url")

# Connecting to PostgreSQL
eng = ce(db_url)

# Loading and viewing the dataset
data = pd.read_sql("select * from diabetes_data;", eng)
print("--------------------------------")
print("Number of rows and columns (Rows, Columns):- ")
print(data.shape)
print("---------------------------------")
print("First 4 rows of unpreprocessed dataset:-")
print(data.head(4))
print("--------------------------------")

# Basic Statistics
print(data.describe())
print("--------------------------------")

# Checking for any missing values
print(data.isnull().sum())
print("--------------------------------")

# Dropping the null values
data.dropna(inplace = True)

# Converting categorical values into numerical values
label_encoder = LabelEncoder()
data["gender"] = data["gender"].str.capitalize() # Normalize casing
valid_genders = ["Male", "Female", "Transgender"]
data = data[data["gender"].isin(valid_genders)] # Drop invalid data

# Encoding the gender
label_encoder_gender = LabelEncoder()
label_encoder_gender.fit(valid_genders)
data["gender"] = label_encoder_gender.transform(data["gender"])
data["smoking_history"] = label_encoder.fit_transform(data["smoking_history"])

# Splitting the dataset into input and output variables
x = data.drop(columns=["diabetes"]) # Input Variable
y = data["diabetes"] # Output Variable

# Handling rare classes : Keeping only the classes which has atleast two samples
counts = y.value_counts()
valid_classes = counts[counts >= 2].index
x = x[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]

# Splitting the dataset into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 50, stratify = y)

# Initializing the model
model = xgbr(
    n_estimators = 1000, 
    learning_rate = 0.05, 
    max_depth = 5, 
    subsample = 0.7, 
    colsample_bytree = 0.8,
    random_state = 56,
    reg_alpha = 0.1,
    reg_lambda = 1.0,
    n_jobs = 1
    )


# Training the model
model.fit(x_train, y_train)

# Evaluating the model
y_pred = model.predict(x_test) # Prediction

# Evaluation Metrics
mse = mse(y_test, y_pred) # Mean Squared Error (MSE)
mae = mae(y_test, y_pred) # Mean Absolute Error (MAE)
r2 = r2(y_test, y_pred) # R^2 Score

# Printing the evaluation metrics
print(f"Mean Squared Error: {mse:.4f}.")
print("--------------------------------")
print(f"Mean Absolute Error: {mae:.4f}.")
print("--------------------------------")
print(f"R^2 Score: {r2:.4f}.")
print("--------------------------------")

# Cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
cv_scores = cross_val_score(model, x, y, cv=cv, scoring='r2')
print(f"Cross-Validation R^2 Scores: {cv_scores}")
print("--------------------------------")
print(f"Average Cross-Validation R^2 Score: {np.mean(cv_scores):.4f}")
print("--------------------------------")

"""# Plotting the evaluation metrics
metrics = {'MSE': mse, 'MAE': mae, 'R2 Score': r2}
plt.figure(figsize=(8,5))
plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green'])
plt.ylabel('Values')
plt.title('Model Evaluation Metrics')
plt.show()

# Plotting the feature distribution
data.hist(bins=30, figsize=(15,10))
plt.suptitle('Feature Distribution')
plt.show()

# Plotting the actual vs predicted values
plt.figure(figsize = (10,6))
plt.scatter(y_test, y_pred, alpha = 0.7)
plt.xlabel("Actual Diabetes Level")
plt.ylabel("Predicted Diabetes Level")
plt.title("Actual vs Predicted Diabetes Level")
plt.plot([0,1], [0,1], color='red', linestyle='--')
plt.show()

# Plotting the feature importance
plt.figure(figsize = (10,6))
plt.barh(x.columns, model.feature_importances_)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Diabetes Prediction")
plt.show()"""

# Saving the model
joblib.dump(model, 'Models/XGBRegressor.pkl')
print("Model saved successfully.")
print("--------------------------------")

# Getting user input for prediction
def get_user_input():
    print("Please provide the following details for diabetes prediction: ")
    print("--------------------------------")

    # Getting user inputs and storing them in a dictionary
    while True:
        gender = input("Enter your gender (M/F/T): ")
        if gender.lower() in ["m", "male"]:
            gender = "Male"
            break
        elif gender.lower() in ["f", "female"]:
            gender = "Female"
            break
        elif gender.lower() in ["t", "transgender"]:
            gender = "Transgender"
            break
        else:
            print("Enter valid gender (M/F/T).")
    
    while True:
        try:
            age = int(input("Enter your age: "))
            if age <= 0 or age > 120:
                print("Age must be a positive integer between 1 and 120. Please enter a valid age.")
            else:
                break
        except ValueError:
            print("Invalid Age Input. Please enter a valid integer for age.")

    while True:
        hypertension = input("Do you have hypertension? (Y/N): ")
        if hypertension.lower() in ["y", "yes"]:
            hypertension = 1
            break
        elif hypertension.lower() in ["n", "no"]:
            hypertension = 0
            break
        else:
            print("Invalid Hypertension Input. Please enter 'Y' or 'N'.")
    
    while True:
        heart_disease = input("Do you have heart disease? (Y/N): ")
        if heart_disease.lower() in ["y", "yes"]:
            heart_disease = 1
            break
        elif heart_disease.lower() in ["n", "no"]:
            heart_disease = 0
            break
        else:
            print("Invalid Heart Disease Input. Please enter 'Y' or 'N'.")
    
    while True:
        smoking_history = input("Do you have a history of smoking? (Former/Current/Never/Can't say): ")
        if smoking_history.lower() not in ["former", "current", "never", "can't say"]:
            print("Enter a valid smoking history. Please enter 'Former' or 'Current' or 'Never' or 'Can't Say'.")
        else:
            smoking_history = smoking_history.lower()
            break
    
    while True:
        bmib = input("Do you know your BMI? (Y/N): ")
        if bmib.lower() == "y" in ["y", "yes"]:
            while True:
                try:
                    bmi = float(input("Enter your BMI: "))
                    if bmi <= 10 or bmi > 70:
                        print("BMI must be a positive number between 10 and 70. Please enter a valid BMI.")
                    else:
                        break
                except ValueError:
                    print("Invalid BMI Input. Please enter a valid decimal number.")
            break
        elif bmib.lower() in ["n", "no"]:
            while True:
                try:
                    weight = float(input("Enter your weight in kg: "))
                    if weight <= 0 or weight > 500:
                        print("Weight must be a positive number between 0 and 500. Please enter a valid weight.")
                    else:
                        break
                except ValueError:
                    print("Invalid weight input. Please enter a valid decimal number.")
            while True:
                try:
                    height = float(input("Enter your height in meters: "))
                    if height <= 0 or height > 3:
                        print("Height must be a positive number between 0 and 3. Please enter a valid height.")
                    else:
                        break
                except ValueError:
                    print("Invalid height input. Please enter a valid decimal number.")
            bmi = round(weight / (height ** 2), 2)   # round immediately
            print(f"Calculated BMI: {bmi}")
            break
        else:
            print("Invalid BMI Input. Please enter 'Y' or 'N'.")
    
    while True:
        try:
            hba1c_level = float(input("Enter your HbA1c level (as a percentage): "))
            if hba1c_level <= 0 or hba1c_level > 20:
                print("HbA1c level must be between 0 and 20. Please enter a valid HbA1c level.")
            else:
                break
        except ValueError:
            print("Invalid HbA1c input. Please enter a valid decimal number.")
    
    user_data = {
        "gender": gender,
        "age": int(age),
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "smoking_history": smoking_history,
        "bmi": float(bmi),
        "hba1c_level": float(hba1c_level),
    }

    # Converting the dictionary into a dataframe
    user_df = pd.DataFrame([user_data])

    # Encode categorical values for prediction
    encoded_df = user_df.copy()
    # Encoding gender
    valid_genders = ["Male", "Female", "Transgender"]
    gender_label_encoder = LabelEncoder()
    gender_label_encoder.fit(valid_genders)
    encoded_df["gender"] = gender_label_encoder.transform(encoded_df["gender"])
    # Encoding smoking history
    valid_smoking_history = ["former", "current", "never", "can't say"]
    smoking_history_label_encoder = LabelEncoder()
    smoking_history_label_encoder.fit(valid_smoking_history)
    encoded_df["smoking_history"] = smoking_history_label_encoder.transform(encoded_df["smoking_history"])

    # Loading the saved model
    loaded_model = joblib.load('Models/XGBRegressor.pkl')
    
    # Prediction
    prediction = loaded_model.predict(encoded_df)[0]
    print(f"The predicted diabetes level is: {prediction:.4f}")
    print("--------------------------------")

    # Printing the final prediction
    if prediction >= 0.5:
        print("The person is likely to have diabetes.")
    else:
        print("The person is unlikely to have diabetes.")
    
    # Storing the prediction in the database
    user_df["diabetes"] = prediction

    try:
        user_df.to_sql("diabetes_data", eng, if_exists = "append", index = False)
        print("The predictions has been saved to the database successfully.")
    except Exception as e:
        print(f"Error while storing prediction: {e}.")

# Prediction loop
if __name__ == "__main__":
    ask = input("Do you want to predict diabetes level? (Y/N): ").strip().lower()
    if ask.lower() in ["y", "yes"]:
        get_user_input()
    elif ask.lower() in ["n", "no"]:
        print("Exiting the program")
        print("--------------------------------")
    else:
        print("Please enter valid diabetes (Y/N).")
        print("--------------------------------")