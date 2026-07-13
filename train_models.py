import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def decisiontree(X_train, X_test, y_train, y_test):
    print("\n==================== Decision Tree Model ====================")
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print(f"Accuracy Score: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    
    return model, y_pred

def randomForest(X_train, X_test, y_train, y_test):
    print("\n==================== Random Forest Model ====================")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print(f"Accuracy Score: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    
    return model, y_pred

def KNN(X_train, X_test, y_train, y_test):
    print("\n==================== KNN Model ====================")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print(f"Accuracy Score: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    
    return model, y_pred

def xgboost_model(X_train, X_test, y_train, y_test):
    print("\n==================== XGBoost Model ====================")
    model = XGBClassifier(n_estimators=50, random_state=42, max_depth=3, learning_rate=0.1, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print(f"Accuracy Score: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    
    return model, y_pred

def compareModel(dt_acc, rf_acc, knn_acc, xgb_acc):
    print("\n==================== Model Comparison ====================")
    print(f"Decision Tree Accuracy : {dt_acc * 100:.2f}%")
    print(f"Random Forest Accuracy : {rf_acc * 100:.2f}%")
    print(f"KNN Accuracy           : {knn_acc * 100:.2f}%")
    print(f"XGBoost Accuracy       : {xgb_acc * 100:.2f}%")
    
    print("\nSelected Model: XGBoost due to stable generalization on structured meteorological parameters.")

def train_and_evaluate():
    # 1. Load the dataset
    dataset_path = "flood_dataset.xlsx"
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Please run prepare_data.py first.")
        return
        
    df = pd.read_excel(dataset_path)
    print("--- Loaded Dataset ---")
    print(df.head())
    print("\nShape:", df.shape)
    
    # 2. Descriptive statistics
    print("\n--- Descriptive Statistics ---")
    print(df.describe())
    
    # 3. Handle Missing Values
    print("\n--- Missing Values Audit ---")
    print(df.isnull().sum())
    
    # 4. Outlier Handling using IQR Capping
    print("\n--- Outlier Capping via IQR ---")
    numeric_cols = ["Annual Rainfall", "Cloud Visibility", "Seasonal Rainfall", "Temperature", "Humidity"]
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers before capping
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        print(f"{col}: Found {outliers} outliers. Capping between {lower_bound:.2f} and {upper_bound:.2f}")
        
        # Perform capping
        df[col] = np.clip(df[col], lower_bound, upper_bound)
        
    # 5. Handle Categorical Values
    print("\n--- Categorical Value Processing ---")
    cloud_cover_map = {"Low": 0, "Medium": 1, "High": 2}
    df["Cloud Cover"] = df["Cloud Cover"].map(cloud_cover_map)
    
    le = LabelEncoder()
    df["class"] = le.fit_transform(df["class"])
    print("Classes mapped:", dict(zip(le.classes_, le.transform(le.classes_))))
    
    # 6. Split X and y
    X = df.drop("class", axis=1)
    y = df["class"]
    
    # Train-test split (80-20, fixed random_state for reproducible split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # 7. Feature Scaling
    print("\n--- Feature Scaling ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the fitted scaler
    joblib.dump(scaler, "transform.save")
    print("Saved StandardScaler as transform.save")
    
    # 8. Train models using global modular functions
    dt_model, dt_pred = decisiontree(X_train_scaled, X_test_scaled, y_train, y_test)
    rf_model, rf_pred = randomForest(X_train_scaled, X_test_scaled, y_train, y_test)
    knn_model, knn_pred = KNN(X_train_scaled, X_test_scaled, y_train, y_test)
    xgb_model, xgb_pred = xgboost_model(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Calculate accuracy metrics
    dt_acc = accuracy_score(y_test, dt_pred)
    rf_acc = accuracy_score(y_test, rf_pred)
    knn_acc = accuracy_score(y_test, knn_pred)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    # Compare
    compareModel(dt_acc, rf_acc, knn_acc, xgb_acc)
    
    # Save selected XGBoost model
    joblib.dump(xgb_model, "floods.save")
    print("\nSaved XGBoost model as floods.save")

if __name__ == "__main__":
    train_and_evaluate()
