import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Memuat model yang telah disimpan
loaded_model = joblib.load('xgb_model.pkl')

# Memuat scaler yang telah disimpan
scaler = joblib.load('scaler.pkl')

# Judul aplikasi
st.title("Klasifikasi Diabetes")

# Input data dari pengguna
Jenis_Kelamin = st.selectbox("Jenis Kelamin", ("L", "P"))
Umur = st.number_input("Umur", min_value=1, max_value=120)
HbA1c = st.number_input("HbA1c", min_value=4.0, max_value=14.0)
Gula_Darah = st.number_input("Gula Darah", min_value=50, max_value=400)

# Proses inputan
if Jenis_Kelamin == "L":
    Jenis_Kelamin = 1
else:
    Jenis_Kelamin = 0

# Buat DataFrame untuk data inputan
new_data = pd.DataFrame({
    'Jenis Kelamin': [Jenis_Kelamin],
    'Umur': [Umur],
    'HbA1c': [HbA1c],
    'Gula Darah': [Gula_Darah]
})

# Memilih kolom numerik untuk normalisasi (kolom yang relevan)
numerical_features = ['Jenis Kelamin', 'Umur', 'HbA1c', 'Gula Darah']

# Melakukan normalisasi pada kolom numerik
new_data[numerical_features] = scaler.transform(new_data[numerical_features])

# Menampilkan data yang sudah dinormalisasi
st.write("Normalized Input Data:", new_data)

# Menghitung probabilitas prediksi
y_prob = loaded_model.predict_proba(new_data)

# Menetapkan threshold (misalnya 0.5)
threshold = 0.5
predictions_based_on_prob = [1 if prob[1] > threshold else 0 for prob in y_prob]

# Mapped hasil prediksi menjadi "Diabetes" atau "Non-Diabetes"
predictions_mapped = ["Diabetes" if pred == 0 else "Non-Diabetes" for pred in predictions_based_on_prob]

# Menampilkan hasil prediksi
st.write(f"Prediction: {predictions_mapped[0]}")