# train.py
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def run_kaggle_pipeline():
    print("🚀 Initializing Kaggle 'Give Me Some Credit' Training Pipeline...")
    
    # Create models directory if it does not exist
    os.makedirs('models', exist_ok=True)
    
    # CHANGED: Look directly in the current folder
    data_path = 'cs-training.csv'
    
    # Safety verification
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Critical Error: Could not find '{data_path}' in this folder. Please ensure your downloaded Kaggle CSV file is named exactly 'cs-training.csv' and sits next to this train.py script.")
        
    print("📥 Loading dataset directly from current folder...")
    df = pd.read_csv(data_path)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    print(f"📊 Dataset successfully loaded. Shape: {df.shape}")
        
    print("🧹 Cleaning missing values...")
    income_median = df['MonthlyIncome'].median()
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(income_median)
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)
    
    features = [
        'age', 
        'MonthlyIncome', 
        'DebtRatio', 
        'RevolvingUtilizationOfUnsecuredLines', 
        'NumberOfTime30-59DaysPastDueNotWorse'
    ]
    
    X = df[features]
    y = df['SeriousDlqin2yrs']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("🏋️ Training balanced Random Forest Classifier (this may take a moment)...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced', 
        max_depth=10,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n📊 ================= KAGGLE VALIDATION METRICS =================")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk (Good)', 'High Risk (Default)']))
    print("================================================================\n")
    
    print("💾 Saving trained models...")
    with open('models/kaggle_rf_model.pkl', 'wb') as m_file:
        pickle.dump(model, m_file)
    with open('models/kaggle_scaler.pkl', 'wb') as s_file:
        pickle.dump(scaler, s_file)
    
    print("✅ Pipeline complete! Model files are saved inside the 'models/' folder.")

if __name__ == '__main__':
    run_kaggle_pipeline()