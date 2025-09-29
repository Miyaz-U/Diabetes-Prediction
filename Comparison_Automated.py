# Importing the necessary libraries
import os
import h2o
import pandas as pd
from sqlalchemy import create_engine as ce
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder as le
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
import joblib
from h2o.automl import H2OAutoML
import warnings
warnings.filterwarnings("ignore")

# Load env and DB
load_dotenv()
db_url = os.getenv("db_url")
eng = ce(db_url)
data = pd.read_sql("SELECT * FROM diabetes_data;", eng)
data.dropna(inplace=True)

# Converting the categorical columns into numeric columns
for col in ["gender", "smoking_history"]:
    if col in data.columns:
        data[col] = le().fit_transform(data[col].astype(str))

# Splitting the dataset into features and target
X = data.drop(columns=["diabetes"]) # Features
y = data["diabetes"] # Target

# Handle rare classes
counts = y.value_counts()
valid_classes = counts[counts >= 2].index
X = X[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]

# Split the dataset into training and testing sets
train_df, test_df = tts(pd.concat([X, y], axis=1), test_size=0.2, random_state=20, stratify=y)
train_df, test_df = train_df.copy(), test_df.copy()
y_col = "diabetes"
x_cols = [col for col in train_df.columns if col != y_col]

# Initialize H2O
h2o.init()
train_h2o = h2o.H2OFrame(train_df)
test_h2o = h2o.H2OFrame(test_df)

# H2O AutoML (Classification)
excluded_algos = ["GBM", "DeepLearning", "StackedEnsemble"]

# Create classification frames
train_h2o_class = h2o.H2OFrame(train_df)
test_h2o_class = h2o.H2OFrame(test_df)
train_h2o_class[y_col] = train_h2o_class[y_col].asfactor()
test_h2o_class[y_col] = test_h2o_class[y_col].asfactor()

aml_class = H2OAutoML(max_runtime_secs=120, seed=20, nfolds=5, exclude_algos=excluded_algos)
aml_class.train(x=x_cols, y=y_col, training_frame=train_h2o_class)
best_h2o_class = aml_class.leader

preds_class = best_h2o_class.predict(test_h2o_class).as_data_frame()["predict"]
if preds_class.dtype in ["float64", "float32"]:
    preds_class = (preds_class > 0.5).astype(int)
acc_score = accuracy_score(test_df[y_col], preds_class)

# H2O AutoML (Regression)
aml_reg = H2OAutoML(max_runtime_secs=120, seed=20, nfolds=5, exclude_algos=excluded_algos)
aml_reg.train(x=x_cols, y=y_col, training_frame=train_h2o)
best_h2o_reg = aml_reg.leader

preds_reg = best_h2o_reg.predict(test_h2o).as_data_frame()["predict"]
r2_score_val = r2_score(test_df[y_col], preds_reg)

# Decision Trees
dtc = DecisionTreeClassifier(random_state=20)
dtc.fit(train_df[x_cols], train_df[y_col])
dtc_preds = dtc.predict(test_df[x_cols])
dtc_score = accuracy_score(test_df[y_col], dtc_preds)

dtr = DecisionTreeRegressor(random_state=20)
dtr.fit(train_df[x_cols], train_df[y_col])
dtr_preds = dtr.predict(test_df[x_cols])
dtr_score = r2_score(test_df[y_col], dtr_preds)

# XGBoost
xgbc = XGBClassifier(random_state=20, use_label_encoder=False, eval_metric='logloss')
xgbc.fit(train_df[x_cols], train_df[y_col])
xgbc_preds = xgbc.predict(test_df[x_cols])
xgbc_score = accuracy_score(test_df[y_col], xgbc_preds)

xgbr = XGBRegressor(random_state=20)
xgbr.fit(train_df[x_cols], train_df[y_col])
xgbr_preds = xgbr.predict(test_df[x_cols])
xgbr_score = r2_score(test_df[y_col], xgbr_preds)

# Collect Results
results = [
    ("H2O_Classifier", best_h2o_class, acc_score, "Accuracy"),
    ("H2O_Regressor", best_h2o_reg, r2_score_val, "R²"),
    ("DecisionTreeClassifier", dtc, dtc_score, "Accuracy"),
    ("DecisionTreeRegressor", dtr, dtr_score, "R²"),
    ("XGBClassifier", xgbc, xgbc_score, "Accuracy"),
    ("XGBRegressor", xgbr, xgbr_score, "R²"),
]

# Map algo to sklearn-style names
algo_map = {
    "XGBoost": {"classification": "XGBClassifier", "regression": "XGBRegressor"},
    "DRF": {"classification": "RandomForestClassifier", "regression": "RandomForestRegressor"},
    "GLM": {"classification": "LogisticRegression", "regression": "LinearRegression"},
}

def translate_algo(h2o_model, problem_type):
    algo = getattr(h2o_model, "algo", None)
    return algo_map.get(algo, {}).get(problem_type, algo)

# Update names for H2O models
results[0] = (translate_algo(best_h2o_class, "classification"), best_h2o_class, acc_score, "Accuracy")
results[1] = (translate_algo(best_h2o_reg, "regression"), best_h2o_reg, r2_score_val, "R²")

# Leaderboard (Sorted)
results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
print("\n📊 Model Comparison Results (Best → Worst):")
for name, _, score, metric in results_sorted:
    print(f"{name}: {metric}={score:.4f}")

# Pick Best
best_model_name, best_model, best_score, best_metric = results_sorted[0]
print(f"\n🏆 Final Best Model: {best_model_name} ({best_metric}={best_score:.4f})")

# Save Best
os.makedirs("Models", exist_ok=True)
if "DecisionTree" in best_model_name or "XGB" in best_model_name:  # sklearn model
    joblib.dump(best_model, "Models/BestModelAutomated.pkl")
else:  # H2O model
    h2o.save_model(best_model, path="Models", force=True, filename="BestModelAutomated")
with open("Models/BestModelAutomated.txt", "w") as f:
    f.write(best_model_name)