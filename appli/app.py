# Importations des outils nécessaires à l'application.
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import normalize

st.set_page_config(page_title="ML Prediction Studio", page_icon="🧬", layout="wide")

# Chemin du dossier principal où se trouvent les modèles déjà entraînés.
BASE_DIR = Path(__file__).resolve().parent.parent

OBESITY_FEATURES = [
    "Gender", "Age", "Height", "Weight", "family_history_with_overweight", "FAVC", "FCVC", "NCP",
    "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS",
]
OBESITY_CATEGORICAL_NAMES = [
    "Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"
]
OBESITY_CATEGORICAL = {}
OBESITY_NUMERIC = [
    "Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE",
]
OBESITY_NAMES = {
    0: "Corpulence normale",
    1: "Obésité classe II",
    2: "Surpoids",
    3: "Obésité classe III",
}
OBESITY_INTERPRETATIONS = {
    0: "IMC moyen d'environ 20,42 — corpulence normale.",
    1: "IMC moyen d'environ 36,52 — obésité de classe II.",
    2: "IMC moyen d'environ 28,11 — situation de surpoids.",
    3: "IMC moyen d'environ 40,89 — obésité de classe III.",
}

CANCER_FEATURES = [
    "radius1", "texture1", "perimeter1", "area1", "smoothness1",
    "compactness1", "concavity1", "concave_points1", "symmetry1", "fractal_dimension1",
    "radius2", "texture2", "perimeter2", "area2", "smoothness2",
    "compactness2", "concavity2", "concave_points2", "symmetry2", "fractal_dimension2",
    "radius3", "texture3", "perimeter3", "area3", "smoothness3",
    "compactness3", "concavity3", "concave_points3", "symmetry3", "fractal_dimension3",
]
CANCER_NAMES = {0: "Bénigne", 1: "Maligne"}
CANCER_INTERPRETATIONS = {
    0: "Diagnostic correspondant à une masse bénigne.",
    1: "Diagnostic correspondant à une masse maligne.",
}

WHEAT_FEATURES = [
    "area A", "perimeter", "compactness", "length of kernel", "width of kernel",
    "asymmetry coefficient", "length of kernel groove",
]
WHEAT_NAMES = {0: "Kama", 1: "Rosa", 2: "Canadian"}

CUSTOMER_FEATURES = [
    "Call  Failure", "Complains", "Subscription  Length", "Charge  Amount",
    "Seconds of Use", "Frequency of use", "Frequency of SMS",
    "Distinct Called Numbers", "Age Group", "Status", "Age", "Customer Value",
]
CUSTOMER_NAMES = {0: "non fidèle", 1: "fidèle"}

# --- Apparence générale de l'application ---

st.markdown("""
<style>
:root { --primary:#2563eb; --ink:#172033; --muted:#667085; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1280px; }
html, body, [data-testid="stAppViewContainer"] { background: #f4f7fb; color: #172033; }
[data-testid="stAppViewContainer"] .main { background: #f4f7fb; }
[data-testid="stHeader"] { background: rgba(244,247,251,.92); }
h1, h2, h3, h4, h5, h6, p, label, span, div { color: inherit; }
[data-testid="stSidebar"] {
  background: #111827;
  border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] * { color: #f8fafc !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #cbd5e1 !important; }
[data-testid="stSidebar"] hr { border-color: #374151; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  background: #1f2937;
  border: 1px solid #374151;
  border-radius: 10px;
  padding: 10px 12px;
  margin-bottom: 8px;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #243044; }
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
[data-testid="stMain"] h4, [data-testid="stMain"] p,
[data-testid="stMain"] label, [data-testid="stMain"] [data-testid="stCaptionContainer"] {
  color: #172033;
}
.hero { background: linear-gradient(135deg,#0f172a 0%,#1d4ed8 100%); color:white;
        padding: 28px 32px; border-radius:18px; margin-bottom:18px; }
.hero h1 { margin:0; font-size:2.05rem; color:#ffffff !important; }
.hero p { margin:8px 0 0; opacity:.88; color:#e2e8f0 !important; }
.card { background:#fff; border:1px solid #e7edf5; border-radius:14px; padding:18px;
        box-shadow:0 8px 24px rgba(15,23,42,.04); }
.result { border-left:5px solid #2563eb; background:#eff6ff; border-radius:12px;
          padding:18px 20px; margin-top:16px; color:#172033; }
.result-danger { border-left:5px solid #dc2626; background:#fef2f2; color:#7f1d1d; }
.result-ok { border-left:5px solid #16a34a; background:#f0fdf4; color:#14532d; }
.result strong, .result span, .small-muted { color: inherit; }
.small-muted { color:#667085; font-size:.9rem; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { padding:10px 18px, color:#475569; }
.stTabs [aria-selected="true"] { color:#2563eb !important; }
[data-testid="stDataFrame"] { background:#ffffff; border-radius:12px; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color:#172033; }
div.stButton > button,
div[data-testid="stDownloadButton"] > button,
button[kind="primary"],
button[kind="secondary"] {
  color: #ffffff !important;
  font-weight: 700;
}
div[data-testid="stDownloadButton"] > button {
  background: #2563eb !important;
  border: 1px solid #2563eb !important;
}
div.stButton > button[kind="primary"] {
  background: #2563eb !important;
  border: 1px solid #2563eb !important;
}
div.stButton > button[kind="primary"]:hover,
div[data-testid="stDownloadButton"] > button:hover {
  background: #1d4ed8 !important;
  border-color: #1d4ed8 !important;
}
div.stButton > button[kind="secondary"] {
  background: #475569 !important;
  border: 1px solid #475569 !important;
}
[data-testid="stAlert"],
[data-testid="stAlert"] * {
  color: #172033 !important;
}
[data-testid="stAlert"] {
  border-radius: 10px;
  border: 1px solid #cbd5e1;
  background: #ffffff !important;
}
[data-testid="stAlert"] [data-baseweb="tag"] {
  color: #172033 !important;
}
div[data-testid="stNotification"],
div[data-testid="stNotification"] * {
  color: #172033 !important;
}
[data-testid="stSuccess"],
[data-testid="stSuccess"] * {
  color: #14532d !important;
  background-color: transparent !important;
}
[data-testid="stError"],
[data-testid="stError"] * {
  color: #7f1d1d !important;
}
[data-testid="stWarning"],
[data-testid="stWarning"] * {
  color: #78350f !important;
}
[data-testid="stInfo"],
[data-testid="stInfo"] * {
  color: #1e3a5f !important;
}
</style>
""", unsafe_allow_html=True)

# --- Chargement des modèles déjà entraînés ---
@st.cache_resource
def load_artifacts():
    obesity_model = joblib.load(BASE_DIR / "model_obeisite.joblib")
    obesity_encoders = joblib.load(BASE_DIR / "encoders_obeisite.joblib")
    obesity_unique_categories = joblib.load(BASE_DIR / "uniques_obeisite.joblib")
    cancer_model = joblib.load(BASE_DIR / "model_cancer.joblib")
    cancer_centroids = joblib.load(BASE_DIR / "centroides_cah_cancer.joblib")
    wheat_model = joblib.load(BASE_DIR / "model_graine.joblib")
    wheat_centroids = joblib.load(BASE_DIR / "centroides_cah_wheat.joblib")
    customer_model = joblib.load(BASE_DIR / "model_cah_customer.joblib")
    customer_centroids = joblib.load(BASE_DIR / "centroides_cah_customer.joblib")
    return (obesity_model, obesity_encoders, obesity_unique_categories, cancer_model,
            cancer_centroids, wheat_model, wheat_centroids, customer_model, customer_centroids)

(obesity_model, obesity_encoders, obesity_unique_categories, cancer_model,
 cancer_centroids, wheat_model, wheat_centroids,
 customer_model, customer_centroids) = load_artifacts()
OBESITY_CATEGORICAL = dict(zip(OBESITY_CATEGORICAL_NAMES, obesity_unique_categories))


def validate_dataframe(df, required_columns, numeric_columns=None):
    # Vérifie les colonnes, les valeurs manquantes et les types numériques du fichier CSV.
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        return False, f"Colonnes manquantes : {', '.join(missing_columns)}", df
    frame = df[required_columns].copy()
    if frame.isna().any().any():
        cols = frame.columns[frame.isna().any()].tolist()
        counts = {c: int(frame[c].isna().sum()) for c in cols}
        return False, "Valeurs manquantes détectées : " + ", ".join(f"{k} ({v})" for k, v in counts.items()), frame
    if numeric_columns:
        bad = []
        for c in numeric_columns:
            converted = pd.to_numeric(frame[c], errors="coerce")
            if converted.isna().any():
                bad.append(c)
            else:
                frame[c] = converted
        if bad:
            return False, "Types numériques invalides : " + ", ".join(bad), frame
    categorical_errors = []
    for col, allowed_values in OBESITY_CATEGORICAL.items():
        if col in frame.columns:
            unknown = sorted(set(frame[col].astype(str)) - set(allowed_values))
            if unknown:
                categorical_errors.append(f"{col} ({', '.join(unknown)})")
    if categorical_errors:
        return False, "Catégories inconnues : " + "; ".join(categorical_errors), frame
    return True, f"{len(frame)} observation(s) valide(s)", frame


def preprocess_obesity(frame):
    encoded = frame.copy()
    categorical_names = [c for c in OBESITY_FEATURES if c in OBESITY_CATEGORICAL]
    for col, encoder in zip(categorical_names, obesity_encoders):
        encoded[col] = encoded[col].astype(str)
        unknown = sorted(set(encoded[col]) - set(encoder.classes_))
        if unknown:
            raise ValueError(f"Valeurs inconnues pour {col} : {', '.join(unknown)}")
        encoded[col] = encoder.transform(encoded[col])
    for col in OBESITY_NUMERIC:
        encoded[col] = pd.to_numeric(encoded[col], errors="raise")
    return encoded[OBESITY_FEATURES].to_numpy(dtype=float)


def predict_obesity(frame):
    values = preprocess_obesity(frame)
    labels = obesity_model.predict(values)
    return labels, [OBESITY_NAMES.get(int(x), f"Classe {int(x)}") for x in labels]


def predict_cancer(frame):
    values = frame[CANCER_FEATURES].to_numpy(dtype=float)
    distances = ((values[:, None, :] - cancer_centroids[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)
    return labels, [CANCER_NAMES.get(int(x), f"Classe {int(x)}") for x in labels]


def predict_wheat(frame):
    values = frame[WHEAT_FEATURES].to_numpy(dtype=float)
    distances = ((values[:, None, :] - wheat_centroids[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)
    return labels, [WHEAT_NAMES.get(int(x), f"Classe {int(x)}") for x in labels]


def predict_customer(frame):
    values = frame[CUSTOMER_FEATURES].to_numpy(dtype=float)
    normalized_values = normalize(values)
    distances = ((normalized_values[:, None, :] - customer_centroids[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)
    return labels, [CUSTOMER_NAMES.get(int(x), f"Classe {int(x)}") for x in labels]


# --- Interfaces Simple Prediction et Multiple Prediction ---

def simple_obesity():
    st.subheader("Prédiction pour un individu")
    st.caption("Saisissez les caractéristiques. Les valeurs sont prétraitées exactement comme celles du fichier CSV.")
    values = {}
    c1, c2, c3 = st.columns(3)
    with c1:
        values["Gender"] = st.selectbox("Sexe", OBESITY_CATEGORICAL["Gender"])
        values["Age"] = st.number_input("Âge", 10.0, 100.0, 25.0, 1.0)
        values["Height"] = st.number_input("Taille (m)", 1.0, 2.5, 1.70, 0.01)
        values["Weight"] = st.number_input("Poids (kg)", 20.0, 300.0, 70.0, 0.5)
        values["family_history_with_overweight"] = st.selectbox("Antécédents familiaux de surpoids", ["no", "yes"])
        values["FAVC"] = st.selectbox("Consommation fréquente d'aliments très caloriques", ["no", "yes"])
    with c2:
        values["FCVC"] = st.slider("Fréquence de consommation de légumes", 1.0, 3.0, 2.0, 0.1)
        values["NCP"] = st.slider("Nombre de repas principaux", 1.0, 4.0, 3.0, 0.1)
        values["CAEC"] = st.selectbox("Alimentation entre les repas", OBESITY_CATEGORICAL["CAEC"])
        values["SMOKE"] = st.selectbox("Fumeur", ["no", "yes"])
        values["CH2O"] = st.slider("Eau consommée par jour (L)", 1.0, 3.0, 2.0, 0.1)
    with c3:
        values["SCC"] = st.selectbox("Surveillance de l'apport calorique", ["no", "yes"])
        values["FAF"] = st.slider("Activité physique (jours/semaine)", 0.0, 3.0, 1.0, 0.1)
        values["TUE"] = st.slider("Temps devant les écrans (heures/jour)", 0.0, 3.0, 1.0, 0.1)
        values["CALC"] = st.selectbox("Consommation d'alcool", OBESITY_CATEGORICAL["CALC"])
        values["MTRANS"] = st.selectbox("Moyen de transport", OBESITY_CATEGORICAL["MTRANS"])

    if st.button("Prédire", type="primary", use_container_width=True):
        frame = pd.DataFrame([[values[c] for c in OBESITY_FEATURES]], columns=OBESITY_FEATURES)
        try:
            labels, names = predict_obesity(frame)
            interpretation = OBESITY_INTERPRETATIONS.get(int(labels[0]), "")
            st.markdown(f'<div class="result"><strong>Classe prédite : {names[0]}</strong><br><span class="small-muted">{interpretation}</span></div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(str(exc))


def multiple_obesity():
    st.subheader("Prédiction sur plusieurs individus")
    st.caption("Importez un CSV contenant exactement les 16 variables nécessaires dans n'importe quel ordre.")
    uploaded = st.file_uploader("Fichier CSV", type=["csv"], key="obesity_csv")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Impossible de lire le fichier : {exc}"); return
        valid, message, frame = validate_dataframe(df, OBESITY_FEATURES, OBESITY_NUMERIC)
        if not valid:
            st.error(message); return
        st.success(message)
        st.dataframe(frame, use_container_width=True)
        if st.button("Prédire toutes les observations", type="primary", key="obesity_predict_many"):
            try:
                labels, names = predict_obesity(frame)
                result = df.copy()
                result["Classe prédite"] = names
                result["Interprétation"] = [OBESITY_INTERPRETATIONS.get(int(x), "") for x in labels]
                st.dataframe(result, use_container_width=True)
                st.download_button("Télécharger les résultats", result.to_csv(index=False).encode("utf-8"),
                                   "resultats_obesite.csv", "text/csv", key="obesity_download")
            except Exception as exc:
                st.error(str(exc))


def simple_cancer():
    st.subheader("Prédiction pour un patient")
    st.caption("Les 30 caractéristiques doivent être mesurées selon les unités du dataset Wisconsin.")
    defaults = load_demo_cancer()
    values = {}
    with st.expander("Saisir les 30 caractéristiques", expanded=True):
        for i in range(0, 30, 3):
            cols = st.columns(3)
            for j, col_name in enumerate(CANCER_FEATURES[i:i+3]):
                with cols[j]:
                    values[col_name] = st.number_input(col_name, value=float(defaults[col_name]), format="%.5f")
    if st.button("Prédire", type="primary", use_container_width=True, key="cancer_predict_one"):
        frame = pd.DataFrame([values], columns=CANCER_FEATURES)
        try:
            labels, names = predict_cancer(frame)
            interpretation = CANCER_INTERPRETATIONS.get(int(labels[0]), "")
            is_malignant = names[0] == "Maligne"
            css = "result-danger" if is_malignant else "result-ok"
            st.markdown(f'<div class="result {css}"><strong>Classe prédite : {names[0]}</strong><br><span class="small-muted">{interpretation}</span></div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(str(exc))


def load_demo_cancer():
    representative_values = cancer_centroids.mean(axis=0)
    return dict(zip(CANCER_FEATURES, representative_values))


def multiple_cancer():
    st.subheader("Prédiction sur plusieurs patients")
    st.caption("Importez un CSV contenant les 30 caractéristiques du diagnostic mammaire.")
    uploaded = st.file_uploader("Fichier CSV", type=["csv"], key="cancer_csv")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Impossible de lire le fichier : {exc}"); return
        valid, message, frame = validate_dataframe(df, CANCER_FEATURES, CANCER_FEATURES)
        if not valid:
            st.error(message); return
        st.success(message)
        st.dataframe(frame, use_container_width=True)
        if st.button("Prédire toutes les observations", type="primary", key="cancer_predict_many"):
            try:
                labels, names = predict_cancer(frame)
                result = df.copy()
                result["Classe prédite"] = names
                result["Interprétation"] = [CANCER_INTERPRETATIONS.get(int(x), "") for x in labels]
                st.dataframe(result, use_container_width=True)
                st.download_button("Télécharger les résultats", result.to_csv(index=False).encode("utf-8"),
                                   "resultats_cancer.csv", "text/csv", key="cancer_download")
            except Exception as exc:
                st.error(str(exc))


def simple_wheat():
    st.subheader("Prédiction pour une graine")
    st.caption("Saisissez les sept caractéristiques morphologiques de la graine.")
    defaults = {
        "area A": 15.26, "perimeter": 14.84, "compactness": 0.871,
        "length of kernel": 5.763, "width of kernel": 3.312,
        "asymmetry coefficient": 2.221, "length of kernel groove": 5.220,
    }
    values = {}
    cols = st.columns(2)
    for i, feature in enumerate(WHEAT_FEATURES):
        with cols[i % 2]:
            values[feature] = st.number_input(feature, value=float(defaults[feature]), format="%.5f")
    if st.button("Prédire", type="primary", use_container_width=True, key="wheat_predict_one"):
        frame = pd.DataFrame([values], columns=WHEAT_FEATURES)
        try:
            labels, names = predict_wheat(frame)
            st.markdown(f'<div class="result result-ok"><strong>Variété prédite : {names[0]}</strong></div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(str(exc))


def multiple_wheat():
    st.subheader("Prédiction sur plusieurs graines")
    st.caption("Importez un CSV contenant les sept caractéristiques du dataset Wheat Seeds.")
    uploaded = st.file_uploader("Fichier CSV", type=["csv"], key="wheat_csv")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Impossible de lire le fichier : {exc}"); return
        valid, message, frame = validate_dataframe(df, WHEAT_FEATURES, WHEAT_FEATURES)
        if not valid:
            st.error(message); return
        st.success(message)
        st.dataframe(frame, use_container_width=True)
        if st.button("Prédire toutes les observations", type="primary", key="wheat_predict_many"):
            try:
                labels, names = predict_wheat(frame)
                result = df.copy()
                result["Variété prédite"] = names
                st.dataframe(result, use_container_width=True)
                st.download_button("Télécharger les résultats", result.to_csv(index=False).encode("utf-8"),
                                   "resultats_graine.csv", "text/csv", key="wheat_download")
            except Exception as exc:
                st.error(str(exc))


def simple_customer():
    st.subheader("Prédiction pour un client")
    st.caption("Saisissez les 12 caractéristiques. Les nouvelles entrées sont normalisées avec la méthode normalize().")
    defaults = {
        "Call  Failure": 7.627937, "Complains": 0.076508,
        "Subscription  Length": 32.541905, "Charge  Amount": 0.942857,
        "Seconds of Use": 4472.459683, "Frequency of use": 69.460635,
        "Frequency of SMS": 73.174921, "Distinct Called Numbers": 23.509841,
        "Age Group": 2.826032, "Status": 1.248254, "Age": 30.998413,
        "Customer Value": 470.972916,
    }
    values = {}
    cols = st.columns(3)
    for i, feature in enumerate(CUSTOMER_FEATURES):
        with cols[i % 3]:
            values[feature] = st.number_input(feature, min_value=0.0,
                                               value=float(defaults[feature]), format="%.4f")
    if st.button("Prédire", type="primary", use_container_width=True, key="customer_predict_one"):
        frame = pd.DataFrame([values], columns=CUSTOMER_FEATURES)
        try:
            labels, names = predict_customer(frame)
            css = "result-ok" if names[0] == "fidèle" else "result-danger"
            st.markdown(f'<div class="result {css}"><strong>Client prédit : {names[0]}</strong></div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(str(exc))


def multiple_customer():
    st.subheader("Prédiction sur plusieurs clients")
    st.caption("Importez un CSV contenant les 12 caractéristiques du dataset Customer Segmentation.")
    uploaded = st.file_uploader("Fichier CSV", type=["csv"], key="customer_csv")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Impossible de lire le fichier : {exc}"); return
        valid, message, frame = validate_dataframe(df, CUSTOMER_FEATURES, CUSTOMER_FEATURES)
        if not valid:
            st.error(message); return
        st.success(message)
        st.dataframe(frame, use_container_width=True)
        if st.button("Prédire tous les clients", type="primary", key="customer_predict_many"):
            try:
                labels, names = predict_customer(frame)
                result = df.copy()
                result["Classe prédite"] = names
                st.dataframe(result, use_container_width=True)
                st.download_button("Télécharger les résultats", result.to_csv(index=False).encode("utf-8"),
                                   "resultats_customer.csv", "text/csv", key="customer_download")
            except Exception as exc:
                st.error(str(exc))


st.markdown('<div class="hero"><h1>ML Prediction Studio</h1><p>Obesity, Breast Cancer, Wheat Seeds & Customer Segmentation</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Modèles")
    model = st.radio("Choisir un modèle",
                     ["Obesity Prediction", "Breast Cancer Prediction",
                      "Wheat Seeds Prediction", "Customer Segmentation"],
                     label_visibility="collapsed")
    st.markdown("---")

# --- Navigation entre les quatre modèles ---
if model == "Obesity Prediction":
    tab1, tab2 = st.tabs(["🔹 Simple Prediction", "📊 Multiple Prediction"])
    with tab1:
        simple_obesity()
    with tab2:
        multiple_obesity()
elif model == "Breast Cancer Prediction":
    tab1, tab2 = st.tabs(["🔹 Simple Prediction", "📊 Multiple Prediction"])
    with tab1:
        simple_cancer()
    with tab2:
        multiple_cancer()
elif model == "Wheat Seeds Prediction":
    tab1, tab2 = st.tabs(["🔹 Simple Prediction", "📊 Multiple Prediction"])
    with tab1:
        simple_wheat()
    with tab2:
        multiple_wheat()
else:
    tab1, tab2 = st.tabs(["🔹 Simple Prediction", "📊 Multiple Prediction"])
    with tab1:
        simple_customer()
    with tab2:
        multiple_customer()
