import io
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Laser Tattoo Removal DSS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Paths and persistent local database
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    BASE_DIR / "models" / "laser_dss_model_bundle.joblib",
    BASE_DIR / "model_bundle.joblib",
    BASE_DIR / "models" / "model_bundle.joblib",
]
DB_PATH = BASE_DIR / "data" / "laser_dss_records.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load trained model bundle
# The model is downloaded from GitHub Release because it is too large for normal GitHub upload.
# ============================================================
import requests

MODEL_URL = "https://github.com/saraalshaheen/laser-tattoo-removal-dss/releases/download/v1.0/laser_dss_model_bundle.joblib"
MODEL_PATH = BASE_DIR / "models" / "laser_dss_model_bundle.joblib"

@st.cache_resource(show_spinner=False)
def load_bundle():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        with st.spinner("Downloading trained model bundle from GitHub Release..."):
            response = requests.get(MODEL_URL, stream=True, timeout=300)
            response.raise_for_status()

            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    return joblib.load(MODEL_PATH)

def normalize_target_name(name):
    name = str(name)
    mapping = {
        "Pulse_Duration": "Pulse Duration",
        "pulse_duration": "Pulse Duration",
        "pulse_duration_ns": "Pulse Duration",
        "Total_Pulses": "Total Pulses",
        "total_pulses": "Total Pulses",
        "Wavelength": "Wavelength",
        "Energy": "Energy",
    }
    return mapping.get(name, name.replace("_", " "))

try:
    bundle = load_bundle()
    loaded_model_path = MODEL_PATH

    metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}

    # The training code saved the final models in selected_models.
    selected_models = bundle.get("selected_models", {}) if isinstance(bundle, dict) else {}

    MODELS = {}
    model_features = None

    for target, obj in selected_models.items():
        target_ui = normalize_target_name(target)

        if isinstance(obj, dict):
            estimator = (
                obj.get("estimator")
                or obj.get("pipeline")
                or obj.get("model")
                or obj.get("best_estimator")
            )
            if model_features is None:
                model_features = obj.get("features")
        else:
            estimator = obj

        if estimator is not None:
            MODELS[target_ui] = estimator

    # Fallback for alternative bundle structures
    if not MODELS and isinstance(bundle, dict) and "models" in bundle:
        for target, estimator in bundle["models"].items():
            MODELS[normalize_target_name(target)] = estimator

    if not MODELS:
        st.error("The model bundle was loaded, but no trained models were found inside it.")
        st.stop()

    FEATURES = (
        metadata.get("main_features")
        or metadata.get("features")
        or model_features
        or [
            "skin_type",
            "tattoo_color",
            "tattoo_type",
            "size_cm2",
            "tattoo_age_years",
            "laser_type_sanitized",
            "repetition_rate_hz",
        ]
    )

    # Options used by the interface
    OPTIONS = {
        "skin_type": metadata.get("skin_type_options", []),
        "colors": metadata.get("tattoo_color_options", metadata.get("color_options", [])),
        "tattoo_type": metadata.get("tattoo_type_options", []),
        "laser_type": metadata.get("laser_type_options", []),
    }

    TARGETS = list(MODELS.keys())
    TARGET_ALIASES = {
        "Pulse_Duration": "Pulse Duration",
        "Total_Pulses": "Total Pulses",
        "Pulse Duration": "Pulse Duration",
        "Total Pulses": "Total Pulses",
        "Wavelength": "Wavelength",
        "Energy": "Energy",
    }

    # Optional metrics table if available
    METRICS = []
    if isinstance(bundle, dict):
        if "final_metrics_df" in bundle:
            try:
                METRICS = bundle["final_metrics_df"].to_dict("records")
            except Exception:
                METRICS = []
        elif "metrics" in bundle:
            METRICS = bundle["metrics"]

except Exception as e:
    st.error("تعذر تحميل حزمة النموذج المدرّب.")
    st.info(f"الملف المتوقع: {MODEL_PATH}")
    st.error("لم يتم العثور على حزمة النموذج أو حدث خطأ أثناء تحميلها من GitHub Release.")
    st.exception(e)
    st.stop()

# ============================================================
# SQLite helpers
# ============================================================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            file_number TEXT,
            age INTEGER,
            gender TEXT,
            notes TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            session_number INTEGER,
            skin_type TEXT,
            colors TEXT,
            size_cm2 REAL,
            tattoo_age_years REAL,
            tattoo_type TEXT,
            laser_type TEXT,
            repetition_rate_hz REAL,
            wavelength REAL,
            energy REAL,
            pulse_duration REAL,
            total_pulses REAL,
            created_at TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
        """
    )
    conn.commit()
    conn.close()

init_db()

def add_patient(full_name, file_number, age, gender, notes):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO patients(full_name, file_number, age, gender, notes, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (full_name, file_number, age, gender, notes, datetime.now().strftime("%Y-%m-%d %H:%M")),
    )
    conn.commit()
    conn.close()

def get_patients():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM patients ORDER BY id DESC", conn)
    conn.close()
    return df

def get_patient(patient_id):
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM patients WHERE id=?", conn, params=(patient_id,))
    conn.close()
    if df.empty:
        return None
    return df.iloc[0].to_dict()

def add_session(patient_id, session_number, row, preds):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sessions(
            patient_id, session_number, skin_type, colors, size_cm2, tattoo_age_years,
            tattoo_type, laser_type, repetition_rate_hz, wavelength, energy,
            pulse_duration, total_pulses, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            session_number,
            row.get("skin_type"),
            row.get("colors"),
            float(row.get("size_cm2", 0)),
            float(row.get("tattoo_age_years", 0)),
            row.get("tattoo_type"),
            row.get("laser_type"),
            float(row.get("repetition_rate_hz", 0)),
            float(preds.get("Wavelength", np.nan)),
            float(preds.get("Energy", np.nan)),
            float(preds.get("Pulse Duration", np.nan)),
            float(preds.get("Total Pulses", np.nan)),
            datetime.now().strftime("%Y-%m-%d %H:%M"),
        ),
    )
    conn.commit()
    conn.close()

def get_sessions(patient_id=None):
    conn = get_conn()
    if patient_id is None:
        df = pd.read_sql_query(
            """
            SELECT s.*, p.full_name, p.file_number
            FROM sessions s LEFT JOIN patients p ON s.patient_id=p.id
            ORDER BY s.id DESC
            """,
            conn,
        )
    else:
        df = pd.read_sql_query(
            """
            SELECT s.*, p.full_name, p.file_number
            FROM sessions s LEFT JOIN patients p ON s.patient_id=p.id
            WHERE s.patient_id=? ORDER BY s.id DESC
            """,
            conn,
            params=(patient_id,),
        )
    conn.close()
    return df

# ============================================================
# Text
# ============================================================
TEXT = {
    "English": {
        "title": "Clinical Laser Tattoo Removal",
        "subtitle": "AI-Powered Device-Aware Decision Support System",
        "dashboard": "Dashboard",
        "patients": "Patients",
        "new_prediction": "New Prediction",
        "about": "About",
        "patient_profile": "Patient Profile",
        "tattoo_characteristics": "Tattoo Characteristics",
        "device_inputs": "Device-Aware Inputs",
        "patient_information": "Patient Information",
        "treatment_sessions": "Treatment Sessions",
        "full_name": "Full Name",
        "file_number": "File Number",
        "age": "Age",
        "gender": "Gender",
        "notes": "Notes",
        "add_patient": "Add Patient",
        "save_patient": "Save Patient",
        "search": "Search by name or file number",
        "view_history": "View History",
        "back_to_patients": "Back to Patients",
        "start_new_session": "Start New Session",
        "session_number": "Session Number",
        "skin_type": "Skin Type (Fitzpatrick Scale)",
        "tattoo_age": "Tattoo Age",
        "years": "Years",
        "months": "Months",
        "tattoo_size": "Tattoo Size (cm²)",
        "tattoo_type": "Tattoo Type",
        "ink_color": "Ink Color",
        "laser_type": "Laser Type",
        "repetition_rate": "Repetition Rate (Hz)",
        "generate": "Generate Treatment Plan",
        "recommended": "Recommended Laser Parameters",
        "wavelength": "Wavelength",
        "energy": "Energy",
        "pulse_duration": "Pulse Duration",
        "total_pulses": "Total Pulses",
        "save_to_record": "Save to Patient Record",
        "download_report": "Download Prediction Report",
        "clinical_disclaimer": "Clinical Disclaimer",
        "no_patients": "No patients saved yet.",
        "no_sessions": "No treatment sessions recorded yet.",
        "total_patients": "Total Patients",
        "total_sessions": "Total Sessions",
        "recent_sessions": "Recent Sessions",
        "enter_patient_name": "Please enter patient name.",
        "patient_saved": "Patient saved successfully.",
        "session_saved": "Session saved to patient record.",
        "select_patient": "Select Patient",
        "disclaimer": "These AI-generated recommendations are intended to assist clinical decision-making and should not replace professional medical judgment. All treatment parameters must be verified by a qualified dermatologist or laser specialist before application.",
        "about_text": "This web-based prototype is connected to the trained machine-learning models developed in the thesis. It receives patient, tattoo, and device-aware inputs, then predicts Wavelength, Energy, Pulse Duration, and Total Pulses as guideline-support outputs. The machine-learning models run in the background; the clinical user sees only interpretable treatment recommendations.",
    },
    "العربية": {
        "title": "النظام السريري لإزالة الوشم بالليزر",
        "subtitle": "نظام ذكي مساعد لاتخاذ القرار يعتمد على خصائص المريض والوشم والجهاز",
        "dashboard": "لوحة التحكم",
        "patients": "المرضى",
        "new_prediction": "تنبؤ جديد",
        "about": "حول النظام",
        "patient_profile": "ملف المريض",
        "tattoo_characteristics": "خصائص الوشم",
        "device_inputs": "مدخلات الجهاز",
        "patient_information": "معلومات المريض",
        "treatment_sessions": "جلسات العلاج",
        "full_name": "الاسم الكامل",
        "file_number": "رقم الملف",
        "age": "العمر",
        "gender": "الجنس",
        "notes": "ملاحظات",
        "add_patient": "إضافة مريض",
        "save_patient": "حفظ المريض",
        "search": "البحث بالاسم أو رقم الملف",
        "view_history": "عرض السجل",
        "back_to_patients": "الرجوع إلى المرضى",
        "start_new_session": "بدء جلسة جديدة",
        "session_number": "رقم الجلسة",
        "skin_type": "نوع البشرة",
        "tattoo_age": "عمر الوشم",
        "years": "سنوات",
        "months": "أشهر",
        "tattoo_size": "حجم الوشم (سم²)",
        "tattoo_type": "نوع الوشم",
        "ink_color": "لون الوشم",
        "laser_type": "نوع الليزر",
        "repetition_rate": "معدل التكرار (Hz)",
        "generate": "توليد خطة العلاج",
        "recommended": "المعاملات المقترحة لليزر",
        "wavelength": "الطول الموجي",
        "energy": "الطاقة",
        "pulse_duration": "مدة النبضة",
        "total_pulses": "عدد النبضات",
        "save_to_record": "حفظ في سجل المريض",
        "download_report": "تحميل تقرير التنبؤ",
        "clinical_disclaimer": "تنبيه سريري",
        "no_patients": "لا توجد بيانات مرضى محفوظة حتى الآن.",
        "no_sessions": "لا توجد جلسات علاج محفوظة حتى الآن.",
        "total_patients": "عدد المرضى",
        "total_sessions": "عدد الجلسات",
        "recent_sessions": "آخر الجلسات",
        "enter_patient_name": "يرجى إدخال اسم المريض.",
        "patient_saved": "تم حفظ المريض بنجاح.",
        "session_saved": "تم حفظ الجلسة في سجل المريض.",
        "select_patient": "اختيار المريض",
        "disclaimer": "هذه التوصيات الناتجة من النظام الذكي مخصصة لدعم القرار السريري فقط، ولا تُعد بديلاً عن رأي الطبيب أو اختصاصي الليزر. يجب مراجعة جميع معاملات العلاج قبل التطبيق العملي.",
        "about_text": "هذا النموذج الأولي يعمل كتطبيق ويب مرتبط بنماذج التعلم الآلي التي تم تدريبها ضمن الرسالة. يستقبل النظام خصائص المريض والوشم والجهاز، ثم يتنبأ بالطول الموجي والطاقة ومدة النبضة وعدد النبضات كمخرجات إرشادية. تعمل النماذج في الخلفية ولا تظهر للمستخدم إلا التوصيات القابلة للفهم السريري.",
    },
}

# ============================================================
# CSS
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"] {font-family: Inter, sans-serif;}
.stApp {background:#f6fbfb;}
.block-container {padding-top:1.2rem; padding-left:3.1rem; padding-right:3.1rem; max-width:1180px;}
section[data-testid="stSidebar"] {background:#edf6f6; border-right:1px solid #d7e5e5;}
section[data-testid="stSidebar"] > div {padding-top:1.1rem;}
.sidebar-logo {display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;}
.logo-icon {background:#005c68;color:white;width:32px;height:32px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-weight:900;}
.logo-text {color:#062f37;font-weight:850;font-size:1.05rem;}
.user-box {position:fixed;bottom:18px;left:18px;width:220px;display:flex;gap:10px;align-items:center;color:#06343d;font-size:.85rem;}
.user-avatar {width:38px;height:38px;border-radius:50%;background:#e8ffff;border:1px solid #cfe4e4;display:flex;align-items:center;justify-content:center;font-weight:850;}
.page-title {color:#073b44;font-weight:900;font-size:2.05rem;line-height:1.15;margin-bottom:.2rem;}
.page-subtitle {color:#63757a;font-size:1.03rem;margin-bottom:.45rem;}
.chip {display:inline-block;padding:.25rem .68rem;border-radius:999px;border:1px solid #cddede;background:#edf5f5;color:#527176;font-size:.78rem;margin-right:.35rem;margin-top:.15rem;}
.chip-active {display:inline-block;padding:.25rem .68rem;border-radius:999px;border:1px solid #82cbd1;background:#d5f7fa;color:#005c68;font-size:.78rem;margin-right:.35rem;margin-top:.15rem;}
.card {background:white;border:1px solid #d7e4e4;border-radius:18px;padding:1.45rem 1.55rem;margin-top:1.15rem;box-shadow:0 2px 8px rgba(0,0,0,.035);}
.section-title {color:#073b44;font-weight:850;font-size:1.05rem;border-left:4px solid #007887;padding-left:.65rem;margin-bottom:1.35rem;}
.result-card {background:#e9f8ed;border:1.6px solid #9bcdae;border-radius:20px;padding:1.2rem .75rem;text-align:center;min-height:170px;margin-bottom:1rem;}
.result-icon {color:#007044;font-size:1.65rem;margin-bottom:.35rem;}
.result-label {color:#087044;font-weight:900;text-transform:uppercase;letter-spacing:.04rem;font-size:.92rem;}
.result-value {color:#052d3a;font-size:2.35rem;font-weight:900;margin:.22rem 0;}
.result-unit {color:#65777b;font-size:.92rem;}
.disclaimer {background:#fff2d5;border:1px solid #e7cd91;color:#67470d;border-radius:18px;padding:1.05rem 1.2rem;line-height:1.55;margin-top:1.2rem;}
.metric-card {background:white;border:1px solid #d7e4e4;border-radius:18px;padding:1.25rem;box-shadow:0 2px 8px rgba(0,0,0,.035);}
.metric-label {color:#62777b;font-size:1rem;}
.metric-value {color:#073b44;font-weight:900;font-size:2.35rem;}
.patient-card {background:white;border:1px solid #d7e4e4;border-radius:18px;padding:1rem 1.1rem;margin:.75rem 0;box-shadow:0 2px 8px rgba(0,0,0,.035);}
.avatar {width:50px;height:50px;border-radius:50%;background:#d7f7f7;color:#006271;display:inline-flex;align-items:center;justify-content:center;font-weight:900;margin-right:.8rem;}
.muted {color:#718286;font-size:.92rem;}
.primary-wave {background:#d4f6fa;color:#00545f;border-radius:10px;padding:.75rem .85rem;font-weight:650;margin-top:.85rem;}
.dot {width:18px;height:18px;border-radius:50%;display:inline-block;margin-right:.45rem;vertical-align:middle;border:1px solid rgba(0,0,0,.22);}
.age-pill {background:#d1f5f7;color:#073b44;border-radius:9px;padding:.55rem;text-align:center;font-weight:850;margin-top:.55rem;}
.stButton>button {border-radius:12px;min-height:2.55rem;font-weight:800;}
div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label, div[data-testid="stTextInput"] label, div[data-testid="stTextArea"] label {color:#0d5962 !important;font-weight:650 !important;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Helpers
# ============================================================
def safe_options(options, fallback):
    values = [str(x) for x in options if str(x).strip() and str(x).lower() != "nan"]
    return values if values else fallback

def clean_color_options(options):
    defaults = ["black", "blue", "green", "red", "orange", "yellow", "purple", "white", "brown"]
    flattened = []
    for opt in options or []:
        for part in str(opt).lower().replace(";", ",").split(","):
            p = part.strip()
            if p and p not in flattened:
                flattened.append(p)
    for d in defaults:
        if d not in flattened:
            flattened.append(d)
    return flattened

def color_hex(color):
    return {
        "black": "#111111",
        "blue": "#1e73be",
        "green": "#2f8a3d",
        "red": "#d32f2f",
        "orange": "#ef5b0c",
        "yellow": "#f5aa2b",
        "purple": "#7b1fb5",
        "white": "#f7f7f7",
        "brown": "#795548",
    }.get(str(color).lower(), "#777777")

def build_row(skin_type, colors, size_cm2, tattoo_age_years, tattoo_type, laser_type, repetition_rate_hz):
    # Build all expected feature columns exactly as training used
    row = {}
    for f in FEATURES:
        if f == "skin_type": row[f] = skin_type
        elif f in ["colors", "tattoo_color"]: row[f] = colors
        elif f == "size_cm2": row[f] = float(size_cm2)
        elif f == "tattoo_age_years": row[f] = float(tattoo_age_years)
        elif f == "tattoo_type": row[f] = tattoo_type
        elif f in ["laser_type", "laser_type_sanitized"]: row[f] = laser_type
        elif f == "repetition_rate_hz": row[f] = float(repetition_rate_hz)
        else: row[f] = np.nan
    return row

def predict_outputs(row):
    X = pd.DataFrame([row], columns=FEATURES)
    raw_preds = {}
    for target, model in MODELS.items():
        raw_preds[target] = float(model.predict(X)[0])
    # normalize names for UI and database
    preds = {}
    for k, v in raw_preds.items():
        preds[TARGET_ALIASES.get(k, k)] = v
    return preds

def show_result_card(icon, label, value, unit):
    st.markdown(f"""
    <div class="result-card">
        <div class="result-icon">{icon}</div>
        <div class="result-label">{label}</div>
        <div class="result-value">{value}</div>
        <div class="result-unit">{unit}</div>
    </div>
    """, unsafe_allow_html=True)

def export_prediction_report(patient, row, preds, T):
    rows = [["Generated At", datetime.now().strftime("%Y-%m-%d %H:%M")]]
    if patient:
        rows.extend([["Patient", patient.get("full_name", "")], ["File Number", patient.get("file_number", "")]])
    for k, v in row.items(): rows.append([k, v])
    for k, v in preds.items(): rows.append([f"Predicted {k}", v])
    rows.append(["Clinical Disclaimer", T["disclaimer"]])
    df = pd.DataFrame(rows, columns=["Field", "Value"])
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Prediction_Report", index=False)
    return out.getvalue()


# ============================================================
# Session state defaults
# ============================================================
if "view" not in st.session_state:
    st.session_state.view = "dashboard"

if "language" not in st.session_state:
    st.session_state.language = "English"

if "selected_patient_id" not in st.session_state:
    st.session_state.selected_patient_id = None

if "last_input" not in st.session_state:
    st.session_state.last_input = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-logo"><div class="logo-icon">⚡</div><div class="logo-text">Laser DSS</div></div>', unsafe_allow_html=True)
    T_side = TEXT.get(st.session_state.get("language", "English"), TEXT["English"])
    if st.button("▦  " + T_side["dashboard"], use_container_width=True): st.session_state.view = "dashboard"
    if st.button("👥  " + T_side["patients"], use_container_width=True): st.session_state.view = "patients"
    if st.button("⚡  " + T_side["new_prediction"], use_container_width=True):
        st.session_state.view = "prediction"; st.session_state.selected_patient_id = None
    if st.button("ℹ️  " + T_side["about"], use_container_width=True): st.session_state.view = "about"
    st.markdown("---")
    st.session_state.language = st.radio("🌐", ["English", "العربية"], index=0 if st.session_state.language == "English" else 1)
    st.markdown('<div class="user-box"><div class="user-avatar">S</div><div><b>Sury Hop</b><br><span class="muted">suryhop@gmail.com</span></div></div>', unsafe_allow_html=True)

T = TEXT[st.session_state.language]

# ============================================================
# Header
# ============================================================
st.markdown(f'<div class="page-title">{T["title"]}</div><div class="page-subtitle">{T["subtitle"]}</div><span class="chip">Device-Aware ML</span><span class="chip-active">✓ Guideline Support</span><span class="chip">Clinical Review</span>', unsafe_allow_html=True)

patients_df = get_patients()
sessions_df = get_sessions()

# ============================================================
# Pages
# ============================================================
if st.session_state.view == "dashboard":
    st.markdown(f"<h1 style='color:#073b44'>{T['dashboard']}</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{T["total_patients"]}</div><div class="metric-value">{len(patients_df)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{T["total_sessions"]}</div><div class="metric-value">{len(sessions_df)}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="section-title">🕘 {T["recent_sessions"]}</div>', unsafe_allow_html=True)
    if sessions_df.empty:
        st.info(T["no_sessions"])
    else:
        for _, s in sessions_df.head(5).iterrows():
            st.write(f"**Session {int(s.session_number)} — {s.colors}**  \nPatient: {s.full_name} · Wavelength: {round(s.wavelength)} nm · Energy: {round(s.energy, 3)}")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.view == "patients":
    st.markdown(f"<h1 style='color:#073b44'>{T['patients']}</h1>", unsafe_allow_html=True)
    with st.expander("➕ " + T["add_patient"], expanded=False):
        st.markdown(f'<div class="card"><div class="section-title">{T["patient_information"]}</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            full_name = st.text_input(T["full_name"])
            age = st.number_input(T["age"], min_value=0, max_value=120, value=30, step=1)
        with b:
            file_number = st.text_input(T["file_number"])
            gender = st.selectbox(T["gender"], ["Female", "Male", "Not specified"])
        notes = st.text_area(T["notes"])
        if st.button("💾 " + T["save_patient"], use_container_width=True):
            if not full_name.strip(): st.warning(T["enter_patient_name"])
            else:
                add_patient(full_name, file_number, age, gender, notes)
                st.success(T["patient_saved"])
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    search = st.text_input("🔎 " + T["search"])
    view_df = patients_df.copy()
    if search.strip() and not view_df.empty:
        q = search.lower()
        view_df = view_df[view_df.full_name.str.lower().str.contains(q, na=False) | view_df.file_number.astype(str).str.lower().str.contains(q, na=False)]
    if view_df.empty:
        st.info(T["no_patients"])
    else:
        for _, p in view_df.iterrows():
            letter = str(p.full_name)[:1].upper() if str(p.full_name) else "?"
            st.markdown(f'<div class="patient-card"><span class="avatar">{letter}</span><b>{p.full_name}</b><br><span class="muted">File: {p.file_number}</span><br><span class="muted">Age: {p.age}</span><br><span class="muted">{p.created_at}</span></div>', unsafe_allow_html=True)
            if st.button("📂 " + T["view_history"], key=f"patient_{p.id}", use_container_width=True):
                st.session_state.selected_patient_id = int(p.id)
                st.session_state.view = "patient_detail"
                st.rerun()
    if not patients_df.empty:
        st.download_button(T["patients"] + " CSV", patients_df.to_csv(index=False).encode("utf-8-sig"), "patients_records.csv", "text/csv", use_container_width=True)

elif st.session_state.view == "patient_detail":
    patient_id = st.session_state.get("selected_patient_id")
    p = get_patient(patient_id) if patient_id else None
    if not p:
        st.warning("No patient selected.")
    else:
        if st.button("← " + T["back_to_patients"]):
            st.session_state.view = "patients"; st.rerun()
        st.markdown(f'<div class="card"><div class="section-title">{T["patient_information"]}</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.write("**" + T["full_name"] + "**"); st.write(p["full_name"])
            st.write("**" + T["age"] + "**"); st.write(p["age"])
        with b:
            st.write("**" + T["file_number"] + "**"); st.write(p["file_number"])
            st.write("**" + T["gender"] + "**"); st.write(p["gender"])
        if p.get("notes"): st.write("**" + T["notes"] + "**"); st.write(p["notes"])
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("⚡ " + T["start_new_session"], use_container_width=True):
            st.session_state.selected_patient_id = int(p["id"])
            st.session_state.view = "prediction"
            st.rerun()
        pdf = get_sessions(p["id"])
        st.markdown(f'<div class="card"><div class="section-title">〰️ {T["treatment_sessions"]}</div>', unsafe_allow_html=True)
        if pdf.empty:
            st.info(T["no_sessions"])
        else:
            for _, s in pdf.iterrows():
                st.write(f"**Session {int(s.session_number)} — {s.colors}**  \nWavelength: {round(s.wavelength)} nm · Energy: {round(s.energy, 3)} · Pulse Duration: {round(s.pulse_duration, 4)} ns · Total Pulses: {round(s.total_pulses)}  \n{s.created_at}")
            st.download_button("Print / Export", pdf.to_csv(index=False).encode("utf-8-sig"), f"patient_{p['file_number']}_sessions.csv", "text/csv", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.view == "prediction":
    selected_patient_id = st.session_state.get("selected_patient_id")
    selected_patient = get_patient(selected_patient_id) if selected_patient_id else None
    if selected_patient:
        st.info(f"{T['new_prediction']} for: {selected_patient['full_name']} | File: {selected_patient['file_number']}")
    skin_options = safe_options(OPTIONS.get("skin_type", []), ["II", "III", "IV", "V"])
    color_options = clean_color_options(OPTIONS.get("colors", []))
    tattoo_type_options = safe_options(OPTIONS.get("tattoo_type", []), ["amateur", "professional"])
    laser_type_options = safe_options(OPTIONS.get("laser_type", []), ["Q-switched Nd:YAG", "Picosecond", "Alexandrite", "Ruby"])
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(f'<div class="card"><div class="section-title">{T["patient_profile"]}</div>', unsafe_allow_html=True)
        skin_type = st.selectbox(T["skin_type"], skin_options)
        st.markdown(f'**{T["tattoo_age"]}**')
        y1, y2 = st.columns(2)
        with y1: tattoo_years = st.number_input(T["years"], 0, 80, 3, 1)
        with y2: tattoo_months = st.number_input(T["months"], 0, 11, 0, 1)
        tattoo_age_years = tattoo_years + tattoo_months / 12.0
        st.markdown(f'<div class="age-pill">{tattoo_age_years:.2f} years</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="card"><div class="section-title">{T["tattoo_characteristics"]}</div>', unsafe_allow_html=True)
        size_cm2 = st.number_input(T["tattoo_size"], min_value=0.1, max_value=1000.0, value=15.0, step=1.0)
        tattoo_type = st.selectbox(T["tattoo_type"], tattoo_type_options)
        color = st.radio(T["ink_color"], color_options, horizontal=True)
        st.markdown(f'<div class="primary-wave"><span class="dot" style="background:{color_hex(color)}"></span>Primary selected color: {color}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="section-title">{T["device_inputs"]}</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1: laser_type = st.selectbox(T["laser_type"], laser_type_options)
    with d2: repetition_rate_hz = st.number_input(T["repetition_rate"], 0.0, 50.0, 5.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)
    row = build_row(skin_type, color, size_cm2, tattoo_age_years, tattoo_type, laser_type, repetition_rate_hz)
    if st.button("⚡ " + T["generate"], use_container_width=True):
        st.session_state.last_input = row
        st.session_state.last_prediction = predict_outputs(row)
    if st.session_state.last_prediction is not None:
        preds = st.session_state.last_prediction
        w = int(round(preds.get("Wavelength", np.nan))) if np.isfinite(preds.get("Wavelength", np.nan)) else "N/A"
        e = round(preds.get("Energy", np.nan), 3) if np.isfinite(preds.get("Energy", np.nan)) else "N/A"
        pdur = round(preds.get("Pulse Duration", np.nan), 4) if np.isfinite(preds.get("Pulse Duration", np.nan)) else "N/A"
        tp = int(round(max(preds.get("Total Pulses", np.nan), 0))) if np.isfinite(preds.get("Total Pulses", np.nan)) else "N/A"
        st.markdown(f'<div class="card"><div class="section-title">{T["recommended"]}</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1: show_result_card("〰️", T["wavelength"], w, "nm")
        with r2: show_result_card("⚡", T["energy"], e, "J/cm² or dataset unit")
        r3, r4 = st.columns(2)
        with r3: show_result_card("⏱️", T["pulse_duration"], pdur, "ns")
        with r4: show_result_card("#", T["total_pulses"], tp, "pulses")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="disclaimer">⚠️ <b>{T["clinical_disclaimer"]}:</b> {T["disclaimer"]}</div>', unsafe_allow_html=True)
        report = export_prediction_report(selected_patient, row, preds, T)
        st.download_button("📄 " + T["download_report"], report, "laser_tattoo_prediction_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        st.markdown(f'<div class="card"><div class="section-title">💾 {T["save_to_record"]}</div>', unsafe_allow_html=True)
        if patients_df.empty:
            st.info(T["no_patients"])
        else:
            labels = [f"{x.full_name} | File: {x.file_number}" for _, x in patients_df.iterrows()]
            ids = [int(x.id) for _, x in patients_df.iterrows()]
            if selected_patient_id and selected_patient_id in ids:
                default_index = ids.index(selected_patient_id)
            else:
                default_index = 0
            selected_label = st.selectbox(T["select_patient"], labels, index=default_index)
            pid = ids[labels.index(selected_label)]
            session_number = st.number_input(T["session_number"], min_value=1, max_value=100, value=1, step=1)
            if st.button("💾 " + T["save_to_record"], use_container_width=True):
                add_session(pid, session_number, row, preds)
                st.success(T["session_saved"])
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.view == "about":
    st.markdown(f'<div class="card"><div class="section-title">{T["about"]}</div>', unsafe_allow_html=True)
    st.write(T["about_text"])
    st.write(f"Model file loaded from: `{loaded_model_path.name}`")
    st.write(f"Number of target outputs: {len(TARGETS)}")
    if METRICS:
        st.write("Internal model summary:")
        st.dataframe(pd.DataFrame(METRICS), use_container_width=True)
    st.markdown(f'<div class="disclaimer">⚠️ <b>{T["clinical_disclaimer"]}:</b> {T["disclaimer"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
