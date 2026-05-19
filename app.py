import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import gdown


model_path = "Models/model (2).pkl"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    gdown.download(url, model_path, quiet=False)

# ── Fix working directory ─────────────────────────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor | India",
    page_icon="🌫️",
    layout="wide"
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
def load_css():
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model  = pickle.load(open('Models/model (2).pkl',         'rb'))
    scaler = pickle.load(open('Models/scaler (2).pkl',        'rb'))
    le     = pickle.load(open('Models/label_encoder.pkl', 'rb'))
    return model, scaler, le

model, scaler, le = load_models()


AQI_DATA = {
    "Good": {
        "color"   : "#00b894",
        "emoji"   : "✅",
        "range"   : "0 - 50",
        "meaning" : "Air quality is satisfactory and poses little or no risk.",
        "risk"    : "Nobody",
        "mask"    : "No mask needed",
        "precautions": {
            "General Public"            : "No precautions needed. Enjoy outdoor activities freely.",
            "Children"                  : "Safe to play outdoors all day.",
            "Elderly"                   : "No restrictions. Good day for a walk.",
            "Heart / Lung Patients"     : "No special precautions needed today.",
            "Athletes / Outdoor Workers": "Ideal conditions. Train outdoors without concern.",
        },
        "do"  : ["Open windows for fresh air", "Exercise outdoors freely", "Let children play outside"],
        "dont": ["Nothing specific to avoid today"],
    },
    "Satisfactory": {
        "color"   : "#f9ca24",
        "emoji"   : "🟡",
        "range"   : "51 - 100",
        "meaning" : "Air quality is acceptable. Minor discomfort for sensitive people.",
        "risk"    : "Very sensitive individuals only",
        "mask"    : "No mask needed. Sensitive groups may use a cloth mask.",
        "precautions": {
            "General Public"            : "Air is fine. Outdoor activities are normal.",
            "Children"                  : "Safe outdoors. Watch for coughing or eye irritation.",
            "Elderly"                   : "Outdoor walks are fine. Avoid dusty areas.",
            "Heart / Lung Patients"     : "Limit strenuous outdoor activity. Carry medication.",
            "Athletes / Outdoor Workers": "Normal activity fine. Take breaks if breathless.",
        },
        "do"  : ["Continue outdoor activities normally", "Stay hydrated", "Monitor sensitive group members"],
        "dont": ["Avoid heavy exercise near busy roads", "Do not ignore breathing discomfort"],
    },
    "Moderate": {
        "color"   : "#e67e22",
        "emoji"   : "🟠",
        "range"   : "101 - 200",
        "meaning" : "Unhealthy for sensitive groups. Others may notice mild effects.",
        "risk"    : "Children, elderly, asthma and heart patients",
        "mask"    : "N95 mask recommended for sensitive groups outdoors.",
        "precautions": {
            "General Public"            : "Reduce prolonged outdoor exertion. Take breaks indoors.",
            "Children"                  : "Limit outdoor play to 1-2 hours. Avoid dusty areas.",
            "Elderly"                   : "Short walks okay. Rest indoors during peak hours.",
            "Heart / Lung Patients"     : "Avoid outdoor exertion. Keep rescue inhaler handy.",
            "Athletes / Outdoor Workers": "Reduce training intensity. Wear N95 during heavy work.",
        },
        "do"  : ["Keep windows closed during peak hours", "Use air purifier indoors", "Drink plenty of water", "Monitor AQI updates"],
        "dont": ["Do not let children play near roads", "Avoid burning garbage", "Do not exercise during rush hours"],
    },
    "Poor": {
        "color"   : "#e74c3c",
        "emoji"   : "🔴",
        "range"   : "201 - 300",
        "meaning" : "Everyone may experience health effects. Sensitive groups severely affected.",
        "risk"    : "Everyone especially children elderly and patients",
        "mask"    : "N95 mask mandatory when going outdoors.",
        "precautions": {
            "General Public"            : "Avoid prolonged outdoor activity. Stay indoors.",
            "Children"                  : "Do not allow outdoor play. Keep indoors at all times.",
            "Elderly"                   : "Stay indoors. Seal window gaps with wet cloth.",
            "Heart / Lung Patients"     : "Do not go outside. Keep emergency medication ready.",
            "Athletes / Outdoor Workers": "Postpone outdoor training. Wear N95 and take breaks.",
        },
        "do"  : ["Stay indoors with windows closed", "Wear N95 mask if going out", "Run air purifier on high", "Seek medical help if chest tightness occurs"],
        "dont": ["No outdoor jogging or cycling", "Do not open windows at peak hours", "Avoid cooking on open flame", "Do not use wood burning stoves"],
    },
    "Very Poor": {
        "color"   : "#6c5ce7",
        "emoji"   : "🟣",
        "range"   : "301 - 400",
        "meaning" : "Health alert. Serious effects for everyone.",
        "risk"    : "Everyone without exception",
        "mask"    : "N95 / N99 mask mandatory even for short outdoor trips.",
        "precautions": {
            "General Public"            : "Stay indoors at all times. Seal doors and windows.",
            "Children"                  : "Schools should suspend outdoor activities. Strictly indoors.",
            "Elderly"                   : "Do not step outside without N95 mask and company.",
            "Heart / Lung Patients"     : "Consider medical consultation. Zero outdoor exposure.",
            "Athletes / Outdoor Workers": "All outdoor work must be suspended with full PPE.",
        },
        "do"  : ["Run HEPA air purifier continuously", "Wet mop floors to settle dust", "Keep emergency contacts ready", "Stay hydrated"],
        "dont": ["Zero outdoor activity", "Do not open windows", "No candles or incense indoors", "Do not use vacuum cleaners"],
    },
    "Severe": {
        "color"   : "#636e72",
        "emoji"   : "⚫",
        "range"   : "401 - 500",
        "meaning" : "Emergency conditions. Entire population seriously affected.",
        "risk"    : "Entire population. Life threatening for vulnerable groups.",
        "mask"    : "N99 / P100 respirator mandatory. Surgical masks are NOT sufficient.",
        "precautions": {
            "General Public"            : "Treat as health emergency. Do not go outside at all.",
            "Children"                  : "Schools must be closed. Children must not leave home.",
            "Elderly"                   : "Hospitalization or shifting to cleaner area recommended.",
            "Heart / Lung Patients"     : "Immediate medical supervision required.",
            "Athletes / Outdoor Workers": "All outdoor work must stop immediately.",
        },
        "do"  : ["Treat this as a public health emergency", "Call emergency services if breathing is difficult", "Seal all gaps with wet towels", "Run multiple air purifiers"],
        "dont": ["Absolutely no outdoor activity", "Do not drive", "No burning of any kind", "Do not ignore respiratory symptoms"],
    },
}

POLLUTANTS = {
    "PM2.5" : {"unit": "ug/m3", "default": 50.0,  "max": 500.0, "safe": 60},
    "PM10"  : {"unit": "ug/m3", "default": 80.0,  "max": 600.0, "safe": 100},
    "NO2"   : {"unit": "ug/m3", "default": 30.0,  "max": 300.0, "safe": 80},
    "SO2"   : {"unit": "ug/m3", "default": 10.0,  "max": 200.0, "safe": 80},
    "CO"    : {"unit": "mg/m3", "default": 1.0,   "max": 50.0,  "safe": 4},
    "O3"    : {"unit": "ug/m3", "default": 20.0,  "max": 200.0, "safe": 100},
}

GROUP_ICONS = {
    "General Public"            : "🧑",
    "Children"                  : "👦",
    "Elderly"                   : "👴",
    "Heart / Lung Patients"     : "🫀",
    "Athletes / Outdoor Workers": "🏃",
}



def result_banner(color, emoji, category, aqi_range, meaning, risk):
    return f"""
    <div style="border-radius:20px; padding:28px 32px; margin:12px 0;
                border:2px solid {color}; background:{color}18;">
        <p style="color:#aaa; margin:0; font-size:0.8rem; letter-spacing:2px;">
            PREDICTION RESULT
        </p>
        <h1 style="color:{color}; margin:8px 0;">
            {emoji} {category}
        </h1>
        <p style="color:#ddd; margin:0;">
            AQI Range: <b>{aqi_range}</b>
        </p>
        <p style="color:#aaa; margin:8px 0 0; font-size:0.9rem;">
            {meaning}
        </p>
        <p style="color:{color}; margin:8px 0 0;
                  font-size:0.85rem; font-weight:600;">
            Who is at risk: {risk}
        </p>
    </div>
    """

def confidence_box(cls, pct, color, is_best):
    weight = "font-weight:700; font-size:1.1rem;" if is_best else ""
    return f"""
    <div style="background:rgba(255,255,255,0.05);
                border:1px solid {color}44;
                border-radius:12px; padding:14px;
                text-align:center; margin:4px 0;">
        <p style="font-size:0.7rem; color:#aaa; margin:0;">
            {cls}
        </p>
        <p style="color:{color}; margin:4px 0; {weight}">
            {pct:.1f}%
        </p>
    </div>
    """

def mask_box(color, mask):
    return f"""
    <div style="border-radius:12px; padding:14px 20px; margin:8px 0;
                background:{color}22; border:1px solid {color};">
        😷 <b>Mask Recommendation:</b>
        <span style="color:{color}; font-weight:600;">
            {mask}
        </span>
    </div>
    """

def precaution_card(color, icon, group, advice):
    return f"""
    <div style="border-radius:14px; padding:18px 22px; margin:8px 0;
                background:rgba(255,255,255,0.04);
                border-left:4px solid {color};">
        <b style="color:{color};">{icon} {group}</b><br>
        <span style="color:#ddd; font-size:0.9rem;">
            {advice}
        </span>
    </div>
    """

def do_card(item):
    return f"""
    <div style="background:#00b89415; border-left:3px solid #00b894;
                border-radius:8px; padding:10px 14px;
                margin:6px 0; color:#ddd; font-size:0.88rem;">
        ✅ {item}
    </div>
    """

def dont_card(item):
    return f"""
    <div style="background:#d6303115; border-left:3px solid #d63031;
                border-radius:8px; padding:10px 14px;
                margin:6px 0; color:#ddd; font-size:0.88rem;">
        ❌ {item}
    </div>
    """

def metric_box(pol, val, limit, color, status):
    return f"""
    <div style="background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.1);
                border-radius:12px; padding:14px;
                text-align:center; margin:4px 0;">
        <p style="font-size:0.75rem; color:#aaa; margin:0;">{pol}</p>
        <p style="font-size:1.3rem; font-weight:700;
                  color:{color}; margin:4px 0;">{val}</p>
        <p style="font-size:0.68rem; color:#888; margin:0;">Limit: {limit}</p>
        <p style="font-size:0.72rem; color:{color}; margin:2px 0;">{status}</p>
    </div>
    """

def reference_card(color, emoji, cat, aqi_range, meaning):
    return f"""
    <div style="border-radius:14px; padding:16px 20px; margin:8px 0;
                background:rgba(255,255,255,0.04);
                border-left:4px solid {color};">
        <b style="color:{color};">{emoji} {cat}</b><br>
        <span style="color:#aaa; font-size:0.82rem;">AQI {aqi_range}</span><br>
        <span style="color:#888; font-size:0.78rem;">{meaning}</span>
    </div>
    """


with st.sidebar:
    st.markdown("## 🌫️ AQI Predictor")
    st.caption("Enter pollutant values to predict AQI and get health precautions.")
    st.divider()

    st.markdown("### Quick Presets")
    col1, col2, col3 = st.columns(3)
    preset = None
    with col1:
        if st.button("Clean"):   preset = [12, 25, 8, 3, 0.3, 45]
    with col2:
        if st.button("Average"): preset = [60, 90, 35, 15, 1.5, 30]
    with col3:
        if st.button("Smoggy"):  preset = [280, 380, 130, 90, 18, 8]

    st.divider()
    st.markdown("### Pollutant Values")

    values = {}
    for i, (name, meta) in enumerate(POLLUTANTS.items()):
        default = float(preset[i]) if preset else meta["default"]
        values[name] = st.number_input(
            f"{name} ({meta['unit']})",
            min_value=0.0,
            max_value=meta["max"],
            value=default,
            step=0.1
        )

    st.divider()
    predict_btn = st.button(
        "Predict AQI + Precautions",
        type="primary",
        use_container_width=True
    )


st.title("AQI Predictor with Health Precautions")
st.caption("Predict Air Quality Index and get personalized health guidance — India")
st.divider()

if predict_btn:

    
    input_df    = pd.DataFrame([values])
    data_scaled = scaler.transform(input_df)
    pred_enc    = model.predict(data_scaled)
    pred_proba  = model.predict_proba(data_scaled)[0]
    category    = le.inverse_transform(pred_enc)[0]
    data        = AQI_DATA[category]
    color       = data["color"]

    
    st.markdown(
        result_banner(color, data["emoji"], category,
                      data["range"], data["meaning"], data["risk"]),
        unsafe_allow_html=True
    )

    
    st.markdown("#### Model Confidence")
    conf_cols = st.columns(len(le.classes_))
    for i, cls in enumerate(le.classes_):
        pct     = pred_proba[i] * 100
        c       = AQI_DATA[cls]["color"]
        is_best = cls == category
        with conf_cols[i]:
            st.markdown(
                confidence_box(cls, pct, c, is_best),
                unsafe_allow_html=True
            )

    st.divider()

    st.markdown(
        mask_box(color, data["mask"]),
        unsafe_allow_html=True
    )

    st.divider()

    
    st.markdown("### Precautions by Group")
    for group, advice in data["precautions"].items():
        icon = GROUP_ICONS.get(group, "👤")
        st.markdown(
            precaution_card(color, icon, group, advice),
            unsafe_allow_html=True
        )

    st.divider()

    
    col_do, col_dont = st.columns(2)
    with col_do:
        st.markdown("### Do's")
        for item in data["do"]:
            st.markdown(do_card(item), unsafe_allow_html=True)

    with col_dont:
        st.markdown("### Don'ts")
        for item in data["dont"]:
            st.markdown(dont_card(item), unsafe_allow_html=True)

    st.divider()

    
    st.markdown("### Your Values vs Safe Limits (CPCB India)")
    p_cols = st.columns(6)
    for i, (pol, val) in enumerate(values.items()):
        limit  = POLLUTANTS[pol]["safe"]
        pct    = val / limit * 100
        col_   = "#00b894" if pct < 80 else ("#f9ca24" if pct < 120 else "#e74c3c")
        status = "Safe"    if pct < 80 else ("Near Limit" if pct < 120 else "Over Limit")
        with p_cols[i]:
            st.markdown(
                metric_box(pol, val, limit, col_, status),
                unsafe_allow_html=True
            )

else:

    
    st.markdown("### Enter values in the sidebar and click Predict")
    st.divider()

    st.markdown("### AQI Category Reference")
    cols = st.columns(3)
    for i, (cat, d) in enumerate(AQI_DATA.items()):
        with cols[i % 3]:
            st.markdown(
                reference_card(d["color"], d["emoji"], cat,
                               d["range"], d["meaning"][:65] + "..."),
                unsafe_allow_html=True
            )

    st.divider()
    st.markdown("### Model Info")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", "Random Forest")
    c2.metric("Accuracy",  "80.26%")
    c3.metric("F1 Score",  "0.80")
    c4.metric("Cities",    "26 Indian Cities")