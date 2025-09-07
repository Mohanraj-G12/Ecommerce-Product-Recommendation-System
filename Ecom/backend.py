# backend.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Load dataset ---
df = pd.read_csv("Ecommerce Dataset.csv")

# --- Clean the data ---
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)  # Remove rows with any missing values

# --- Encode categorical features ---
categoricals = df.select_dtypes(include=["object", "category"]).columns.tolist()
df = pd.get_dummies(df, columns=categoricals, drop_first=True)

# --- Set target column ---
target_col = "Label" if "Label" in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# --- Scale numeric features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- Logistic Regression ---
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_report_text = classification_report(y_test, lr_pred)

# --- K-Nearest Neighbors ---
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_report_text = classification_report(y_test, knn_pred)

# --- Decision Tree ---
dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_report_text = classification_report(y_test, dt_pred)

# --- Random Forest (powerful) ---
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_report_text = classification_report(y_test, rf_pred)

# --- Return results for frontend ---
def get_model_results():
    return {
        "Logistic Regression": {
            "accuracy": lr_accuracy,
            "report_text": lr_report_text
        },
        "K-Nearest Neighbors": {
            "accuracy": knn_accuracy,
            "report_text": knn_report_text
        },
        "Decision Tree": {
            "accuracy": dt_accuracy,
            "report_text": dt_report_text
        },
        "Random Forest": {
            "accuracy": rf_accuracy,
            "report_text": rf_report_text
        }
    }
