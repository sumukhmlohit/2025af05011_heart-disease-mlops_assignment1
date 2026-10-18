import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

def perform_eda_and_cleaning():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("docs/screenshots", exist_ok=True)
    
    print("[INFO] Downloading UCI Heart Disease dataset...")
    df = pd.read_csv(UCI_URL, names=COLUMNS, na_values='?')
    df.dropna(inplace=True)
    df['target'] = (df['target'] > 0).astype(int)
    
    df.to_csv("data/raw/heart.csv", index=False)
    print(f"[INFO] Cleaned dataset saved. Shape: {df.shape}")
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='Set2')
    plt.title("Heart Disease Class Balance Distribution")
    plt.savefig("docs/screenshots/class_balance.png", bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.savefig("docs/screenshots/correlation_heatmap.png", bbox_inches='tight')
    plt.close()
    print("[INFO] EDA plots saved under docs/screenshots/.")

if __name__ == "__main__":
    perform_eda_and_cleaning()
