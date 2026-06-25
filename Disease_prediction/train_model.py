import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_save_model(file_name, target_column, model_name):
    print("\n" + "="*50)
    print(f" Processing Dataset: {file_name}")
    print("="*50)
    
    # Check if file exists before proceeding
    if not os.path.exists(file_name):
        print(f"Warning: '{file_name}' not found. Skipping...")
        return
        
    # Load dataset, explicitly converting '?' strings into true NaN missing values
    df = pd.read_csv(file_name, na_values='?')
    
    # Clean column whitespace
    df.columns = df.columns.str.strip()
    
    # Drop any completely empty or unnamed columns caused by trailing commas
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(how='all', axis=1)
    
    # Verify target column exists
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in {file_name}!")
        print(f"Available columns: {list(df.columns)[:5]}...")
        return
        
    # Handle non-numeric target values (e.g., 'M' and 'B' in breast cancer data)
    if df[target_column].dtype == 'object':
        unique_classes = df[target_column].dropna().unique()
        if len(unique_classes) == 2:
            class_mapping = {unique_classes[0]: 1, unique_classes[1]: 0}
            df[target_column] = df[target_column].map(class_mapping)
            print(f"Encoded target categories automatically: {class_mapping}")
    
    # Safely convert remaining object columns to numeric types
    for col in df.columns:
        if df[col].dtype == 'object' and col != target_column:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # CRITICAL FIX: Convert heart disease labels (0-4) into clean binary classification (0 or 1)
    if model_name == "heart":
        df[target_column] = df[target_column].apply(lambda x: 1 if x > 0 else 0)
        print("--> Dynamic mapping applied: Grouped heart condition severity levels into binary outcome.")
    
    # Impute missing items globally using column medians
    df = df.fillna(df.median(numeric_only=True))
        
    # Feature Extraction
    drop_cols = [target_column]
    if 'id' in df.columns:
        drop_cols.append('id')
        
    X = df.drop(columns=drop_cols)
    y = df[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_accuracy = 0
    best_model = None
    
    # Iterate and train models
    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = clf
            
    print(f"\n--> Best Model chosen for {model_name}: {best_model.__class__.__name__} ({best_accuracy*100:.2f}%)")
    
    # Save model and matching data scaler artifacts
    with open(f'models/{model_name}_model.pkl', 'wb') as m_file:
        pickle.dump(best_model, m_file)
    with open(f'models/{model_name}_scaler.pkl', 'wb') as s_file:
        pickle.dump(scaler, s_file)
        
    print(f"Artifacts saved successfully: 'models/{model_name}_model.pkl' & 'models/{model_name}_scaler.pkl'")

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
        
    datasets_config = [
        {"file": "diabetes.csv", "target": "Outcome", "label": "diabetes"},
        {"file": "cleveland.csv", "target": "num", "label": "heart"}, 
        {"file": "data.csv", "target": "diagnosis", "label": "general"}     
    ]
    
    for config in datasets_config:
        train_and_save_model(config["file"], config["target"], config["label"])
        
    print("\n" + "="*50)
    print(" ALL DATASETS TRAINED AND ARTIFACTS PLACED IN 'models/' ")
    print("="*50)