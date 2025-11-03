import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import os

app = Flask(__name__)

DISPLAY_DISASTER_TYPES = [
    'Lũ lụt', 'Bão/Áp thấp', 'Lũ quét/Sạt lở', 'Hạn hán', 
    'Nắng nóng', 'Rét đậm/Rét hại', 'Xâm nhập mặn', 'Động đất'
]

MODEL_NAME_MAPPING = {
    'Lũ lụt': 'Flood',
    'Bão/Áp thấp': 'Storm',
    'Lũ quét/Sạt lở': 'Landslide', 
    'Hạn hán': 'Drought',
    'Nắng nóng': 'Drought',
    'Rét đậm/Rét hại': 'Drought',
    'Xâm nhập mặn': 'Salinity', 
    'Động đất': 'Storm',
}

ALL_FEATURES = ['Magnitude', 'Latitude', 'Longitude', 'Total Deaths', 'No. Injured', 'No. Affected', 'No. Homeless', 'Total Affected', 'economic_damage_adjusted', 'CPI', 'time', 'Disaster Subgroup', 'Disaster Type', 'Disaster Subtype', 'Location', 'Magnitude Scale', 'River Basin', 'month_year', 'Vung'] 

VIETNAM_REGIONS = ['0', 'Bac Trung Bo', 'Dong Bac Bo', 'Dong Bang Song Cuu Long', 'Dong Bang Song Hong', 'Dong Nam Bộ', 'Nam Trung Bo', 'Tay Bac Bo', 'Tay Nguyen', 'Viet Nam'] # Dùng tên lớp gốc

LOADED_MODELS = {}

try:
    successful_models = set(MODEL_NAME_MAPPING.values())
    for model_base_name in successful_models:
        file_name = f"xgb_risk_{model_base_name}.pkl"
        LOADED_MODELS[model_base_name] = joblib.load(file_name)
    print(f"Successfully loaded {len(LOADED_MODELS)} core models.")

except Exception as e:
    print(f"ERROR: Failed to load models for deployment. Please ensure all 5 required files exist: {e}")
    LOADED_MODELS = None
    
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_results = None
    
    if request.method == "POST" and LOADED_MODELS:
        try:
            input_nam = int(request.form.get("nam"))
            input_thang = int(request.form.get("thang"))
            input_vung = request.form.get("vung")
            
            # 1. Tạo DataFrame GIẢ LẬP ĐẦU VÀO
            input_data = {}
            for col in ALL_FEATURES:
                if col in ['Disaster Subgroup', 'Disaster Type', 'Disaster Subtype', 'Location', 'Magnitude Scale', 'River Basin', 'month_year', 'Vung']:
                    input_data[col] = ['']
                else:
                    input_data[col] = [0]
            
            # Ghi đè các giá trị động
            input_data['time'] = [input_nam + input_thang/12] 
            input_data['month_year'] = [f"{input_thang}/{input_nam}"]
            input_data['Vung'] = [input_vung] # THÊM VÙNG ĐÃ CHỌN VÀO ĐẦU VÀO
            
            X_new = pd.DataFrame(input_data)
            
            dynamic_risks = {}
            
            # 2. CHẠY DỰ ĐOÁN CHO 8 LOẠI THIÊN TAI BẰNG 5 MÔ HÌNH CÓ SẴN
            for display_name in DISPLAY_DISASTER_TYPES:
                model_base_name = MODEL_NAME_MAPPING[display_name]
                model_pipeline = LOADED_MODELS[model_base_name]
                
                # Dự đoán xác suất xảy ra (lớp 1)
                proba_occurrence = model_pipeline.predict_proba(X_new)[0][1] 
                
                dynamic_risks[display_name] = f"{proba_occurrence * 100:.2f}%"
            
            # 3. Kết quả Cuối cùng
            prediction_results = {
                "input_nam": input_nam,
                "input_thang": input_thang,
                "input_vung": input_vung,
                "risks": dynamic_risks
            }
            
        except Exception as e:
            prediction_results = {"error": f"Lỗi dự đoán: {e}"}
            
    return render_template("index.html", results=prediction_results, regions=VIETNAM_REGIONS)

if __name__ == "__main__":
    app.run(debug=True)