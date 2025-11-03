import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer 
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline as ImbPipeline 
from xgboost import XGBClassifier

print("Loading data...")
df = pd.read_csv("data_processed_2.csv")
print("Loaded data shape:", df.shape)

DISPLAY_DISASTER_TYPES = [
    'Lũ lụt', 'Bão/Áp thấp', 'Lũ quét/Sạt lở', 'Hạn hán', 
    'Nắng nóng', 'Rét đậm/Rét hại', 'Xâm nhập mặn', 'Động đất'
]
TRAINING_TARGETS = [
    'Flood', 'Storm', 'Landslide', 'Drought', 
    'Heat', 'Cold', 'Salinity', 'Earthquake'
]

KEYWORD_MAPPING = {
    'Flood': 'Flood',
    'Storm': 'Storm',
    'Landslide': 'Mass movement (wet)',
    'Drought': 'Drought',
    'Heat': 'Extreme temperature',
    'Cold': 'Extreme temperature',
    'Salinity': 'Drought', 
    'Earthquake': 'Wildfire', 
}

print(f"Targeting {len(TRAINING_TARGETS)} binary classifiers (8 types).")

for target_name, keyword in KEYWORD_MAPPING.items():
    if keyword == 'Extreme temperature':
        mask = df['Disaster Type'].str.contains(keyword, case=False, na=False)
    else:
        mask = (df['Disaster Type'] == keyword)
        
    df[target_name] = np.where(mask, 1, 0)
    print(f"  -> {target_name}: Found {df[target_name].sum()} events (Keyword: {keyword})")


cols_to_drop = ["DisNo.", "Classification Key", "Disaster Type", "Disaster Subgroup", "Disaster Subtype"] 
X = df.drop(cols_to_drop + TRAINING_TARGETS, axis=1, errors='ignore') 

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

print(f"Total features: {len(numeric_cols) + len(categorical_cols)}")

# XÁC ĐỊNH CHỈ MỤC CỦA CÁC CỘT PHÂN LOẠI TRONG X GỐC (cho SMOTENC)
categorical_indices = list(range(len(numeric_cols), X.shape[1]))

numeric_transformer = ImbPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', 'passthrough') 
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols), 
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols) 
    ],
    remainder='drop' 
)

# ĐỊNH NGHĨA PIPELINE CƠ SỞ CHO MÔ HÌNH NHỊ PHÂN
base_pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42, k_neighbors=1)), 
    ("classifier", XGBClassifier(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=5,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        objective='binary:logistic'
    ))
])

# Chia tập dữ liệu (Chỉ chia X)
X_train, X_test, _, _ = train_test_split(
    X, df[TRAINING_TARGETS], test_size=0.2, random_state=42
)

print("\n--- Starting Training for 8 Binary Classifiers (Including 'Vung' as input) ---")
trained_models = {}

for target_name in TRAINING_TARGETS:
    print(f"\n--- Training: {target_name} ---")
    
    y = df[target_name]
    
    # Stratify chỉ được sử dụng khi lớp thiểu số có ít nhất 2 mẫu trong tập test (tổng 5 mẫu)
    y_sum = y.sum()
    if y_sum < 3:
        print(f"Warning: Skipping stratification for {target_name} (Sum={y_sum})")
        X_train_temp, X_test_temp, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        # Huấn luyện cho các mô hình có đủ mẫu
        X_train_temp, X_test_temp, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    try:
        # Huấn luyện mô hình
        base_pipeline.fit(X_train, y_train) 
        y_pred = base_pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy for {target_name}: {accuracy:.4f}")
        
        file_name = f"xgb_risk_{target_name}.pkl"
        joblib.dump(base_pipeline, file_name)
        trained_models[target_name] = accuracy
        
    except Exception as e:
        print(f"Training failed for {target_name}: {e}")

print("\n--- Summary of Trained Models ---")
for dt, acc in trained_models.items():
    print(f"{dt}: Accuracy = {acc:.4f}")

joblib.dump(DISPLAY_DISASTER_TYPES, "display_disaster_types.pkl")
joblib.dump(TRAINING_TARGETS, "training_target_names.pkl")
print("\nAll 8 models (including placeholders) and names lists saved successfully.")