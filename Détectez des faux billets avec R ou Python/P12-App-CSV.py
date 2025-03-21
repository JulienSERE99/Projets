 
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Pour charger le modèle

# Charger le modèle de régression logistique
model = joblib.load("log_reg.pkl")
scaler = joblib.load("scaler.pkl")

# Interface utilisateur
st.title("📊 Détection de Faux Billets via CSV")

st.write("📂 **Chargez un fichier CSV contenant les caractéristiques des billets pour analyse.**")

# Upload du fichier CSV
uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Lire le fichier
    df = pd.read_csv(uploaded_file)

    # Vérifier que les colonnes requises sont bien présentes
    expected_columns = ["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]
    
    if all(col in df.columns for col in expected_columns):
        st.write("✅ **Fichier chargé avec succès !**")
        
        df_selected = df[expected_columns]
        df_scaled = scaler.transform(df_selected)

        # Prédire avec le modèle
        predictions = model.predict(df_scaled)

        # Ajouter les prédictions au DataFrame
        df["Prediction"] = ["✅ Authentique" if pred == 1 else "❌ Faux" for pred in predictions]

        # Afficher les résultats
        st.write("### 🏆 Résultats des Prédictions")
        st.dataframe(df)

        # Télécharger les résultats en CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Télécharger les résultats", data=csv, file_name="predictions.csv", mime="text/csv")

    else:
        st.error("🚨 **Le fichier ne contient pas toutes les colonnes requises !**")
        st.write(f"Colonnes attendues : {expected_columns}")
        st.write(f"Colonnes fournies : {list(df.columns)}")
