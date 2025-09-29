# Importing the required modules
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score as r2
from sklearn.metrics import accuracy_score as acc, f1_score as f1, precision_score as ps, recall_score as rs
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import os

load_dotenv()

# Connecting to the database
db_url = os.getenv("db_url")
engine = create_engine(db_url)

# Loading the dataset
data = pd.read_sql("SELECT * FROM diabetes_data;", engine)
data.dropna(inplace=True)

# Converting the Gender column into numerical values
valid_genders = ["Male", "Female", "Transgender"]
gender_le = LabelEncoder()
gender_le.fit(valid_genders)
data["gender"] = data["gender"].str.capitalize()  # normalize
data = data[data["gender"].isin(valid_genders)]   # drop invalid
data["gender"] = gender_le.transform(data["gender"])

# Normalize smoking_history
data["smoking_history"] = data["smoking_history"].str.lower().str.strip()

# Fix common typos / variations
replace_map = {
    "ever": "never",
    "cant say": "can't say",
    "cannot say": "can't say",
    "na": "never",
    "n/a": "never",
}
data["smoking_history"] = data["smoking_history"].replace(replace_map)

# Keeping only the valid categories
valid_smoking = ["former", "current", "never", "can't say"]
data = data[data["smoking_history"].isin(valid_smoking)]

# Converting the Smoking History column into numerical values
smoke_le = LabelEncoder()
smoke_le.fit(valid_smoking)
data["smoking_history"] = smoke_le.transform(data["smoking_history"])


# Separating the columns into input and output columns
x = data.drop(columns=["diabetes"]) # Input Column
y = data["diabetes"] # Output Column

# Handling rare classes : Keeping only the classes which has atleast two samples
counts = y.value_counts()
print(counts)
valid_classes = counts[counts >= 2].index
x = x[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]

# Splitting the datas into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=50, stratify=y
)

# Loading the saved models
models = {
    "XGBRegressor": joblib.load("Models/XGBRegressor.pkl"),
    "XGBClassifier" : joblib.load("Models/XGBClassifier.pkl"),
    "RandomForestClassifier" : joblib.load("Models/RandomForestClassifier.pkl"),
    "RandomForestRegressor" : joblib.load("Models/RandomForestRegressor.pkl"),
    "DecisionTreeClassifier" : joblib.load("Models/DecisionTreeClassifier.pkl"),
    "DecisionTreeRegressor" : joblib.load("Models/DecisionTreeRegressor.pkl"),
    "LinearRegression" : joblib.load("Models/LinearRegression.pkl")
}

# Comparing and printing the best Classifier model
print("==========================================")
print("Classifier Results")
print("------------------------------------------")
classifier_metrics = {} # Dictionary for storing metrics
for cname in ["RandomForestClassifier", "DecisionTreeClassifier", "XGBClassifier"]:
    model = models[cname]
    y_pred_c = model.predict(x_test)
    accs = acc(y_test, y_pred_c)
    f1s = f1(y_test, y_pred_c, average = "weighted")
    prs = ps(y_test, y_pred_c, average = "weighted")
    res = rs(y_test, y_pred_c, average = "weighted")
    classifier_metrics[cname] = accs
    print(f"Accuracy of {cname}: {accs: .4f}.")
    print(f"F1 Score of {cname}: {f1s: .4f}.")
    print(f"Precision Score of {cname}: {prs: .4f}.")
    print(f"Recall Score of {cname}: {res: .4f}.")
    print("------------------------------------------")
best_classifier = max(classifier_metrics, key = classifier_metrics.get)
best_accuracy = classifier_metrics[best_classifier]
print(f"The best classifier model is {best_classifier} with an accuracy score of {best_accuracy: .4f}.")
print("==========================================")

# Comparing and printing the best Regressor model
print("==========================================")
print("Regressor Results")
print("------------------------------------------")
regressor_metrics = {}
for rname in ["XGBRegressor", "RandomForestRegressor", "DecisionTreeRegressor", "LinearRegression"]:
    model = models[rname]
    y_pred_r = model.predict(x_test)
    mesqer = mse(y_test, y_pred_r)
    meaber = mae(y_test, y_pred_r)
    r2s = r2(y_test, y_pred_r)
    regressor_metrics[rname] = r2s
    print(f"Mean Squared Error (MSE) of {rname}: {mesqer: .4f}.")
    print(f"Mean Absolute Error (MAE) of {rname}: {meaber: .4f}.")
    print(f"R2 Score of {rname}: {r2s: .4f}.")
    print("------------------------------------------")
best_regressor = max(regressor_metrics, key = regressor_metrics.get)
best_r2 = regressor_metrics[best_regressor]
print(f"The best regressor model is {best_regressor} with an R2 score of {best_r2: .4f}.")
print("==========================================")

# Comparing the best Classifier model and best Regressor Model and printing the overall best model
print("==========================================")
print("Final Comparison")
print("------------------------------------------")
if best_accuracy >= best_r2:
    print(f"The overall best model is {best_classifier} (Classifier) with an accuracy score of {best_accuracy: .4f}.")
    best_model = best_classifier
    print("==========================================")
else:
    print(f"The overall best model is {best_regressor} (Regressor) with a R2 score of {best_r2: .4f}.")
    best_model = best_regressor
    print("==========================================")

# Saving the best model
joblib.dump(models[best_model], f"Models/BestModelManual.pkl")
print("Best model has been saved successfully.")

# Saving the name of the best model
with open("Models/BestModelManual.txt", "w") as f:
    f.write(best_model)