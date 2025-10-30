import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import tensorflow as tf
import streamlit as st # Import thư viện Streamlit
import os

# ---------------------------------------------------------------------------
# CẤU HÌNH TRANG WEB (Streamlit)
# ---------------------------------------------------------------------------
# Đặt tiêu đề và layout cho trang web
st.set_page_config(
    page_title="Dự báo Rủi ro Thiên tai",
    page_icon="🌊",
    layout="wide"
)

# ---------------------------------------------------------------------------
# TẢI CÁC MÔ HÌNH VÀ ĐỐI TƯỢNG (Giống hệt code Flask)
# ---------------------------------------------------------------------------
# Dùng @st.cache_resource để tải mô hình 1 LẦN DUY NHẤT
@st.cache_resource
def load_models():
    print("--- Đang tải các mô hình và đối tượng ---")
    MODEL_DIR = "models"
    
    try:
        # Tải mô hình LSTM
        lstm_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'lstm_model.keras'))
        
        # Tải mô hình XGBoost
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(os.path.join(MODEL_DIR, 'xgb_model.json'))
        
        # Tải Scaler (LSTM)
        scaler_lstm = joblib.load(os.path.join(MODEL_DIR, 'scaler_lstm.joblib'))
        
        # Tải Preprocessor (XGB)
        preprocessor_xgb = joblib.load(os.path.join(MODEL_DIR, 'preprocessor_xgb.joblib'))
        
        # Tải Label Encoder (XGB)
        encoder_xgb_label = joblib.load(os.path.join(MODEL_DIR, 'encoder_xgb_label.joblib'))
        
        # Tải dữ liệu lịch sử (LSTM)
        ts_data_history_df = joblib.load(os.path.join(MODEL_DIR, 'ts_data_history.joblib'))
        ts_history_values = ts_data_history_df['event_count'].values
        
        # Lấy danh sách các Vùng từ preprocessor để làm dropdown
        VUNG_CATEGORIES = list(preprocessor_xgb.categories_[0])
        
        TIME_STEP = 12 # Phải giống với lúc huấn luyện

        print("--- Tải mô hình thành công ---")
        
        return (lstm_model, xgb_model, scaler_lstm, preprocessor_xgb, 
                encoder_xgb_label, ts_history_values, VUNG_CATEGORIES, TIME_STEP)

    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG: Không thể tải mô hình. Lỗi: {e}")
        st.error(f"Lỗi khi tải mô hình: {e}")
        return (None,) * 8

# Tải tất cả các mô hình
(lstm_model, xgb_model, scaler_lstm, preprocessor_xgb, 
 encoder_xgb_label, ts_history_values, VUNG_CATEGORIES, TIME_STEP) = load_models()

# ---------------------------------------------------------------------------
# CÁC HÀM DỰ BÁO (Giống hệt code Flask)
# ---------------------------------------------------------------------------
def predict_frequency_internal():
    try:
        last_sequence = ts_history_values[-TIME_STEP:]
        scaled_input = scaler_lstm.transform(last_sequence.reshape(-1, 1))
        current_input = scaled_input.reshape((1, TIME_STEP, 1))
        predicted_scaled_value = lstm_model.predict(current_input, verbose=0)
        predicted_value = scaler_lstm.inverse_transform(predicted_scaled_value)
        result = max(0, predicted_value[0][0])
        return result
    except Exception as e:
        return f"Lỗi: {e}"

def predict_disaster_type_internal(latitude, longitude, vung, month):
    try:
        input_df = pd.DataFrame({
            'Latitude': [latitude], 'Longitude': [longitude],
            'Vùng': [vung], 'Month': [month]
        })
        processed_input = preprocessor_xgb.transform(input_df)
        prediction_encoded = xgb_model.predict(processed_input)
        prediction_proba = xgb_model.predict_proba(processed_input)
        predicted_label = encoder_xgb_label.inverse_transform(prediction_encoded)[0]
        probability = np.max(prediction_proba) * 100
        return predicted_label, probability
    except Exception as e:
        return f"Lỗi: {e}", 0

# ---------------------------------------------------------------------------
# XÂY DỰNG GIAO DIỆN WEB (Streamlit)
# ---------------------------------------------------------------------------
st.title("🌊 Hệ thống Phân tích và Dự báo Rủi ro Thiên tai")
st.markdown("Nguyên mẫu Web tích hợp mô hình LSTM và XGBoost (Deploy bằng Streamlit)")

st.divider() # Kẻ vạch ngang

# --- PHẦN 1: DỰ BÁO TẦN SUẤT (LSTM) ---
st.header("📊 Dự báo Tần suất (LSTM)")

# Chạy dự báo LSTM
freq_result = predict_frequency_internal()
if isinstance(freq_result, (int, float, np.number)):
    # Dùng st.metric để hiển thị con số
    st.metric(label="Dự báo tần suất sự kiện trong tháng tới", 
              value=f"{freq_result:.2f} sự kiện")
else:
    st.error(f"Lỗi dự báo tần suất: {freq_result}")

st.divider() # Kẻ vạch ngang

# --- PHẦN 2: DỰ BÁO PHÂN LOẠI (XGBOOST) ---
st.header("🗺️ Phân loại Thiên tai (XGBoost)")
st.markdown("Nhập các thông tin dưới đây để phân loại rủi ro:")

# Tạo 2 cột để layout
col1, col2 = st.columns(2)

with col1:
    lat = st.number_input("Vĩ độ (Latitude)", 
                         format="%.2f", value=17.48, 
                         help="Ví dụ: 17.48 cho Bắc Trung Bộ")
    
    # Dùng st.selectbox cho Vùng
    vung = st.selectbox("Chọn Vùng", 
                        options=VUNG_CATEGORIES, 
                        index=VUNG_CATEGORIES.index("Bắc Trung Bộ")) # Đặt giá trị mặc định

with col2:
    lon = st.number_input("Kinh độ (Longitude)", 
                         format="%.2f", value=106.60, 
                         help="Ví dụ: 106.60 cho Bắc Trung Bộ")
    
    month = st.number_input("Tháng (1-12)", 
                           min_value=1, max_value=12, value=10)

# Nút bấm dự báo
if st.button("Dự báo Phân loại", type="primary"):
    if xgb_model: # Kiểm tra xem mô hình đã tải chưa
        label, proba = predict_disaster_type_internal(lat, lon, vung, month)
        if "Lỗi" in str(label):
            st.error(label)
        else:
            # Hiển thị kết quả
            st.success(f"Dự báo: **{label}**")
            st.info(f"Độ tin cậy: **{proba:.2f}%**")
    else:
        st.error("Mô hình XGBoost chưa được tải. Vui lòng kiểm tra lỗi.")