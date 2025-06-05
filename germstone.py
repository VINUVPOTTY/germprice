import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Set page config
st.set_page_config(page_title="Gemstone Price Predictor", layout="centered")

# Load and prepare data silently
@st.cache_data
def load_and_train_model():
    df = pd.read_csv("gemstone.csv")

    if 'id' in df.columns:
        df.drop(['id'], axis=1, inplace=True)

    # Outlier removal
    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # Encode categoricals
    dummies = pd.get_dummies(df[['cut', 'color', 'clarity']], drop_first=True, dtype=int)
    df_encoded = pd.concat([dummies, df.drop(['cut', 'color', 'clarity'], axis=1)], axis=1)
    df_encoded.drop(['table', 'depth'], axis=1, inplace=True, errors='ignore')

    X = df_encoded.drop('price', axis=1)
    y = df_encoded['price']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return df, model, scaler, X.columns

# Load everything
df, model, scaler, column_order = load_and_train_model()

# UI Inputs only
st.markdown("""
    <h1 style="
        text-align: center;
        font-size: 3em;
        background: linear-gradient(90deg, #FF5F6D, #FFC371);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        animation: pulse 2s infinite;
    ">
        💎 Gemstone Price Predictor
    </h1>
    <style>
    @keyframes pulse {
        0% { text-shadow: 0 0 5px #ffc371; }
        50% { text-shadow: 0 0 25px #ff5f6d; }
        100% { text-shadow: 0 0 5px #ffc371; }
    }
    </style>
""", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 1.2em; color: #FFFFFF;'>"
    "Predict the price of gemstones based on their features and quality attributes."
    "</p>",
    unsafe_allow_html=True
)

st.markdown('<h4 style="color: white;">Enter Gemstone Features</h4>', unsafe_allow_html=True)


st.markdown('<p style="color: white; font-size: 20px;">Carat</p>', unsafe_allow_html=True)
carat = st.number_input("", 0.0, 5.0, 1.0)

st.markdown('<p style="color: white; font-size: 20px;">Length (x)</p>', unsafe_allow_html=True)
x = st.number_input("", 0.0, 10.0, 5.5)

st.markdown('<p style="color: white; font-size: 20px;">Width (y)</p>', unsafe_allow_html=True)
y_val = st.number_input("", 0.0, 10.0, 5.6)

st.markdown('<p style="color: white; font-size: 20px;">Depth (z)</p>', unsafe_allow_html=True)
z = st.number_input("", 0.0, 10.0, 3.5)

st.markdown('<h4 style="color: white;">Select Quality Attributes</h4>', unsafe_allow_html=True)

cut = st.selectbox("Cut", sorted(df['cut'].unique()))
color = st.selectbox("Color", sorted(df['color'].unique()))
clarity = st.selectbox("Clarity", sorted(df['clarity'].unique()))

# Build prediction input
input_data = pd.DataFrame([{'carat': carat, 'x': x, 'y': y_val, 'z': z}])
categorical = pd.get_dummies(pd.DataFrame([[cut, color, clarity]], columns=['cut', 'color', 'clarity']), drop_first=True, dtype=int)
final_input = pd.concat([categorical, input_data], axis=1)
final_input = final_input.reindex(columns=column_order, fill_value=0)
final_scaled = scaler.transform(final_input)

# Predict
if st.button(" Predict Price"):
    pred_price = model.predict(final_scaled)[0]
    st.markdown(
        f"<div style='background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;'>"
        f"<h3 style='color:#333;'>💰 Predicted Gemstone Price:</h3>"
        f"<h1 style='color:#4CAF50;'>${pred_price:,.2f}</h1>"
        f"</div>",
        unsafe_allow_html=True
    )

import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image path
add_bg_from_local('diamond.png')