import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def handle_missing_values(df):
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

def encode_categorical(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    return df

def scale_features(df):
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if 'Churn' in num_cols:
        num_cols = num_cols.drop('Churn')
    
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    input_path = "../dataset_raw/data_churn.csv"
    output_path = "dataset_preprocessing/data_clean.csv"

    df = load_data(input_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = encode_categorical(df)
    df = scale_features(df)

    save_data(df, output_path)

    print("Preprocessing selesai! Data disimpan di:", output_path)

if __name__ == "__main__":
    main()