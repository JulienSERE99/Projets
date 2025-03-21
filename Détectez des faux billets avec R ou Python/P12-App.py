
import streamlit as st
import pandas as pd
import numpy as np
import joblib  

# Charger le modèle 
model = joblib.load("log_reg.pkl")
scaler = joblib.load("scaler.pkl")

# Interface utilisateur
st.title("💵 Détection de Faux Billets")
st.write("Entrez les dimensions du billet pour vérifier s'il est authentique.")

# Champs d'entrée pour les dimensions du billet
diagonal = st.number_input ("Diagonal", min_value=0.0, step=1.0)
height_left = st.number_input("Height Left", min_value=0.0, step=1.0)
height_right = st.number_input("Height Right", min_value=0.0, step=1.0)
margin_low = st.number_input("Margin Low", min_value=0.0, step=1.0)
margin_up = st.number_input("Margin Up", min_value=0.0, step=1.0)
length = st.number_input("Length", min_value=0.0, step=1.0)

# Bouton pour faire la prédiction
if st.button("🔍 Vérifier le billet"):
    # Préparer les données sous forme de DataFrame
    input_data = pd.DataFrame([{
    "diagonal" : diagonal,
        "height_left": height_left,
        "height_right": height_right,
        "margin_low": margin_low,
        "margin_up": margin_up,
        "length": length
    }])

    # Appliquer la même normalisation que lors de l'entraînement
    input_data_scaled = scaler.transform(input_data)

    # Prédire si le billet est vrai ou faux
    prediction = model.predict(input_data_scaled)

    # Afficher le résultat
    if prediction[0] == 1:
        st.success("✅ Le billet est AUTHENTIQUE !")
    else:
        st.error("❌ Le billet est FAUX !")
