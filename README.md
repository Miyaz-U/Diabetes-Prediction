# Diabetes Prediction System

This project provides multiple machine learning approaches to predict diabetes, including Logistic Regression, Decision Tree, Random Forest, XGBoost, and H2O AutoML.  
It also includes automated and manual model comparison scripts, along with a Streamlit-based web application.

## 📂 Project Structure
- **Diabetes_Predictor_LR.py** → Logistic Regression model  
- **Diabetes_Predictor_DTC.py** → Decision Tree Classifier  
- **Diabetes_Predictor_DTR.py** → Decision Tree Regressor  
- **Diabetes_Predictor_RFC.py** → Random Forest Classifier  
- **Diabetes_Predictor_RFR.py** → Random Forest Regressor  
- **Diabetes_Predictor_XGBC.py** → XGBoost Classifier  
- **Diabetes_Predictor_XGBR.py** → XGBoost Regressor  
- **Diabetes_Predictor_App_Manual_Comparison.py** → Manual model comparison  
- **Diabetes_Predictor_App_Automated_Comparison.py** → Automated model comparison  
- **Comparison_Manual.py / Comparison_Automated.py** → Scripts for performance benchmarking  

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
For example, to run the Streamlit diabetes predictor app:
```bash
streamlit run Diabetes_Predictor_App_Manual_Comparison.py
```

## 🛠 Dependencies
- numpy  
- pandas  
- matplotlib  
- scikit-learn  
- xgboost  
- h2o  
- joblib  
- streamlit  
- sqlalchemy  
- python-dotenv  

## 📖 Notes
- Requires **Python 3.8+**.  
- Ensure your `.env` file is properly configured if using database connections.  
- Models can be trained and compared using both manual and automated approaches.  

---

### 📌 Author
Developed for diabetes prediction and model performance benchmarking using classical ML and AutoML approaches.
