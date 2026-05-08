import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Prediksi Workout",
    page_icon="🏋️",
    layout="centered"
)

model  = joblib.load("workout_model.pkl")
scaler = joblib.load("scaler_nb.pkl")

VALID_RANGE = {
    "Durasi Workout (jam)":     (0.5,  2.0),
    "Calories Burned":          (300, 1500),
    "Rata-rata BPM (Avg BPM)": (120,  170),
    "BPM Maksimal (Max BPM)":  (160,  200),
}

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: #f4f7fb;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none !important;
}

[data-testid="stAppViewContainer"] > .main > .block-container {
    background: white;
    border-radius: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    padding: 40px 50px 50px 50px !important;
    margin-top: 40px;
    margin-bottom: 40px;
    max-width: 700px;
}

.emoji-center {
    text-align: center;
    font-size: 70px;
    margin-bottom: 5px;
}

.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #1e3c72;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #4b5563;
    font-size: 18px;
    margin-bottom: 25px;
}

label {
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

.stTextInput > div > div > input {
    border-radius: 12px;
    border: 1.5px solid #cbd5e1;
    padding: 12px 14px;
    font-size: 17px;
    color: #111827;
    background: #f8fafc;
}

.stTextInput > div > div > input:focus {
    border-color: #3B7597;
    background: white;
}

.stTextInput > div > div > input::placeholder {
    color: #9ca3af;
    font-weight: 400;
}

.stButton > button {
    width: 100%;
    border-radius: 15px;
    height: 52px;
    border: none;
    background: #3B7597;
    color: white;
    font-size: 18px;
    font-weight: bold;
    margin-top: 8px;
    transition: all 0.3s;
}

.stButton > button:hover {
    background: #2d5c78 !important;
    transform: scale(1.02);
    box-shadow: 0 5px 15px rgba(0,0,0,0.15);
}

.result-box {
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
    background: #3B7597;
    color: #FFFFFF;
}

.error-box {
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    font-size: 15px;
    font-weight: 600;
    margin-top: 15px;
    background: #fee2e2;
    color: #b91c1c;
}

.warning-box {
    padding: 14px;
    border-radius: 12px;
    font-size: 14px;
    font-weight: 500;
    margin-top: 15px;
    margin-bottom: 5px;
    background: #fef9c3;
    color: #854d0e;
    border-left: 4px solid #eab308;
}

.range-hint {
    font-size: 12px;
    color: #94a3b8;
    margin-top: -8px;
    margin-bottom: 12px;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu,
header {
    display: none !important;
    visibility: hidden !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="emoji-center">🏋️‍♂️</div>', unsafe_allow_html=True)
st.markdown('<div class="title">Prediksi Intensitas Workout</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Masukkan data workout untuk mengetahui intensitas latihan kamu</div>', unsafe_allow_html=True)

duration_str = st.text_input("⏱️ Durasi Workout (jam)", placeholder="Contoh: 1.5")
st.markdown('<div class="range-hint">Range valid: 0.5 – 2.0 jam</div>', unsafe_allow_html=True)

calories_str = st.text_input("🔥 Calories Burned", placeholder="Contoh: 800")
st.markdown('<div class="range-hint">Range valid: 300 – 1500 kalori</div>', unsafe_allow_html=True)

avg_bpm_str = st.text_input("💓 Rata-rata BPM (Avg BPM)", placeholder="Contoh: 140")
st.markdown('<div class="range-hint">Range valid: 120 – 170 BPM</div>', unsafe_allow_html=True)

max_bpm_str = st.text_input("❤️‍🔥 BPM Maksimal (Max BPM)", placeholder="Contoh: 175")
st.markdown('<div class="range-hint">Range valid: 160 – 200 BPM</div>', unsafe_allow_html=True)

if st.button("Prediksi Sekarang"):
    try:
        duration = float(duration_str.replace(",", "."))
        calories = float(calories_str.replace(",", "."))
        avg_bpm  = float(avg_bpm_str.replace(",", "."))
        max_bpm  = float(max_bpm_str.replace(",", "."))

        inputs = {
            "Durasi Workout (jam)":     duration,
            "Calories Burned":          calories,
            "Rata-rata BPM (Avg BPM)": avg_bpm,
            "BPM Maksimal (Max BPM)":  max_bpm,
        }

        out_of_range = []
        for field, value in inputs.items():
            low, high = VALID_RANGE[field]
            if not (low <= value <= high):
                out_of_range.append(f"<b>{field}</b>: {value} (valid: {low}–{high})")

        if out_of_range:
            warning_html = "⚠️ <b>Input di luar range data training, hasil prediksi mungkin tidak akurat:</b><br>" + "<br>".join(out_of_range)
            st.markdown(f'<div class="warning-box">{warning_html}</div>', unsafe_allow_html=True)

        raw         = np.array([[duration, calories, avg_bpm, max_bpm]])
        data_scaled = scaler.transform(raw)
        prediction  = model.predict(data_scaled)
        hasil       = prediction[0]

        if hasil == "Rendah":
            emoji = "🔴"
            text  = "Intensitas Rendah"
        elif hasil == "Sedang":
            emoji = "🟡"
            text  = "Intensitas Sedang"
        else:
            emoji = "🟢"
            text  = "Intensitas Tinggi"

        st.markdown(
            f'<div class="result-box">{emoji} {text}</div>',
            unsafe_allow_html=True
        )

    except ValueError:
        st.markdown(
            '<div class="error-box">⚠️ Harap masukkan angka yang valid di semua kolom.</div>',
            unsafe_allow_html=True
        )