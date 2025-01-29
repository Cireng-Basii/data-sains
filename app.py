import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the model
model_filename = 'model_regresi_narkotika_terbaik.sav'  # Ganti jika nama file berbeda
try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: Model file '{model_filename}' not found. Please make sure the model file exists.")
    st.stop()

# Function for preprocessing input (sesuaikan dengan data preparation yang dilakukan)
def preprocess_input(data):
    df = pd.DataFrame([data])

    # 1. Encoding (gunakan OneHotEncoder)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    #  Pastikan 'Jenis Zat' yang diinputkan ada di dalam list
    if 'Jenis Zat' in df.columns:
        # Fit dan transformasikan data training
        X_encoded = encoder.fit_transform(df[['Jenis Zat']])
        # Buat DataFrame baru dengan kolom yang telah di-encode
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['Jenis Zat']))

        # Gabungkan kembali dengan kolom numerik asli
        df = df.drop('Jenis Zat', axis=1)
        df = pd.concat([df, X_encoded_df], axis=1)
    
        # 3. Feature Scaling (Standardization)
        scaler = StandardScaler()
        numerical_cols = df.select_dtypes(include=np.number).columns
        #numerical_cols = numerical_cols.drop('Total')  # 'Total' tidak ada di data input
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # 4. Transformasi Data (PowerTransformer)
        pt = PowerTransformer(method='yeo-johnson')
        df[numerical_cols] = pt.fit_transform(df[numerical_cols])

        # 5. Handle Missing Values
        imputer = SimpleImputer(strategy='median')
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

        return df
    else:
        st.error("Error: 'Jenis Zat' column not found in input data.")
        return pd.DataFrame()  # Return an empty DataFrame to avoid further errors

# Streamlit app
st.title('Prediksi Total Penyalahguna Narkotika')

# Input form
jenis_zat = st.selectbox('Jenis Zat', ['GANJA', 'SHABU', 'EKSTASI', 'HEROIN', 'COCAINE', 'METHADONE', 'MORFIN', 'KODEIN', 'MDMA', 'BUPRENORFIN', 'KETAMIN', 'NPS'])  # Sesuaikan dengan jenis zat yang ada
jml_rawat_jalan = st.number_input('Jumlah Rawat Jalan', min_value=0, value=0)
jml_ibm = st.number_input('Jumlah IBM', min_value=0, value=0)
jml_rawat_inap = st.number_input('Jumlah Rawat Inap', min_value=0, value=0)

# Create input data dictionary
input_data = {
    'Jenis Zat': jenis_zat,
    'Jumlah Rawat Jalan': jml_rawat_jalan,
    'Jumlah IBM': jml_ibm,
    'Jumlah Rawat Inap': jml_rawat_inap
}

# Predict button
if st.button('Prediksi'):
    # Preprocess input
    input_df = preprocess_input(input_data)

    # Make prediction
    if not input_df.empty:
        prediction = model.predict(input_df)
        st.success(f'Prediksi Total Penyalahguna: {prediction[0]:,.2f}')