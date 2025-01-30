import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the model
model_filename = 'model_regresi_narkotika_terbaik.sav'
try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: Model file '{model_filename}' not found.")
    st.stop()

# Fungsi preprocessing yang lebih robust
def preprocess_input(data):
    df = pd.DataFrame([data])
    
    # 1. Cek dan handle kolom konstan
    numerical_cols = df.select_dtypes(include=np.number).columns
    constant_cols = df[numerical_cols].columns[df[numerical_cols].var() == 0]
    df = df.drop(columns=constant_cols)
    
    # 2. Handle nilai <= 0 untuk PowerTransformer
    for col in numerical_cols:
        if (df[col] <= 0).any():
            df[col] = df[col] + 1e-6
    
    # 3. Definisikan pipeline preprocessing
    categorical_features = ['Jenis Zat']
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('power', PowerTransformer(method='yeo-johnson')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    try:
        processed_data = preprocessor.fit_transform(df)
        return pd.DataFrame(processed_data, columns=preprocessor.get_feature_names_out())
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return pd.DataFrame()

# Streamlit UI
st.title('Prediksi Total Penyalahguna Narkotika')

# Input form
jenis_zat = st.selectbox('Jenis Zat', ['GANJA', 'SHABU', 'EKSTASI', 'HEROIN', 'COCAINE', 
                                      'METHADONE', 'MORFIN', 'KODEIN', 'MDMA', 'BUPRENORFIN',
                                      'KETAMIN', 'NPS'])
jml_rawat_jalan = st.number_input('Jumlah Rawat Jalan', min_value=0, value=0)
jml_ibm = st.number_input('Jumlah IBM', min_value=0, value=0)
jml_rawat_inap = st.number_input('Jumlah Rawat Inap', min_value=0, value=0)

# Prediction logic
if st.button('Prediksi'):
    input_data = {
        'Jenis Zat': jenis_zat,
        'Jumlah Rawat Jalan': jml_rawat_jalan,
        'Jumlah IBM': jml_ibm,
        'Jumlah Rawat Inap': jml_rawat_inap
    }
    
    input_df = preprocess_input(input_data)
    
    if not input_df.empty:
        try:
            prediction = model.predict(input_df)
            st.success(f'Prediksi Total Penyalahguna: {prediction[0]:,.2f}')
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
