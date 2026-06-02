
import io
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Laser Tattoo Removal DSS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_URL = "https://github.com/saraalshaheen/laser-tattoo-removal-dss/releases/download/v1.0/laser_dss_model_bundle.joblib"
MODEL_PATH = BASE_DIR / "models" / "laser_dss_model_bundle.joblib"
DB_PATH = BASE_DIR / "data" / "laser_dss_records.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load trained model bundle
# ============================================================
@st.cache_resource(show_spinner=False)
def load_bundle():
    if not MODEL_PATH.exists():
        with st.spinner("Downloading trained model bundle..."):
            response = requests.get(MODEL_URL, stream=True, timeout=300)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    return joblib.load(MODEL_PATH)

try:
    bundle = load_bundle()
except Exception as e:
    st.error("تعذر تحميل حزمة النموذج المدرّب.")
    st.info(f"الملف المتوقع: {MODEL_PATH}")
    st.error("لم يتم العثور على حزمة النموذج أو حدث خطأ أثناء تحميلها من GitHub Release.")
    st.exception(e)
    st.stop()

# ============================================================
# Robust bundle parsing
# ============================================================
metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}

def normalize_target_name(name):
    """Return the display name used by the UI regardless of how the target was saved."""
    raw = str(name).strip()
    key = raw.lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "wavelength": "Wavelength",
        "wavelength_nm": "Wavelength_nm",
        "energy": "Energy",
        "energy_jcm2": "Energy_Jcm2",
        "energy_j_cm2": "Energy_Jcm2",
        "pulse_duration": "Pulse Duration",
        "pulseduration": "Pulse Duration",
        "pulse_duration_ns": "Pulse Duration",
        "total_pulses": "Total Pulses",
        "totalpulses": "Total Pulses",
        "total_pulse": "Total Pulses",
    }
    return aliases.get(key, raw)

def extract_models(bundle):
    models = {}
    if isinstance(bundle, dict):
        if "selected_models" in bundle and isinstance(bundle["selected_models"], dict):
            for target, obj in bundle["selected_models"].items():
                if isinstance(obj, dict):
                    est = (
                        obj.get("estimator")
                        or obj.get("model")
                        or obj.get("pipeline")
                        or obj.get("best_estimator")
                        or obj.get("best_model")
                        or obj.get("selected_model")
                        or obj.get("trained_model")
                    )
                else:
                    est = obj
                if est is not None:
                    models[normalize_target_name(target)] = est
        elif "models" in bundle and isinstance(bundle["models"], dict):
            for target, est in bundle["models"].items():
                models[normalize_target_name(target)] = est
        elif "target_models" in bundle and isinstance(bundle["target_models"], dict):
            for target, est in bundle["target_models"].items():
                models[normalize_target_name(target)] = est
    return models

MODELS = extract_models(bundle)

if not MODELS:
    st.error("لم يتم العثور على النماذج داخل ملف model bundle.")
    st.write("Available bundle keys:", list(bundle.keys()) if isinstance(bundle, dict) else type(bundle))
    st.stop()

TARGETS = list(MODELS.keys())

def extract_features(bundle, metadata, models):
    candidate_keys = [
        "features", "feature_columns", "input_features", "X_columns",
        "model_features", "training_features"
    ]
    for key in candidate_keys:
        if isinstance(bundle, dict) and key in bundle and bundle[key]:
            return list(bundle[key])
        if isinstance(metadata, dict) and key in metadata and metadata[key]:
            return list(metadata[key])

    # Try to infer from any pipeline/estimator
    for model in models.values():
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        if hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
    # Safe default used in final training pipeline
    return [
        "skin_type",
        "tattoo_color",
        "size_cm2",
        "tattoo_age_years",
        "tattoo_type",
        "laser_type_sanitized",
        "repetition_rate_hz",
    ]

FEATURES = extract_features(bundle, metadata, MODELS)

def extract_options(metadata):
    opts = {}
    if isinstance(metadata, dict):
        # direct options dictionary
        if "options" in metadata and isinstance(metadata["options"], dict):
            opts.update(metadata["options"])
        # common metadata fields
        for key in ["skin_type", "tattoo_color", "colors", "tattoo_type", "laser_type", "laser_type_sanitized"]:
            for candidate in [key, key + "_options"]:
                if candidate in metadata and metadata[candidate]:
                    opts[key] = metadata[candidate]
        # keys used by earlier app versions
        if "skin_type_options" in metadata:
            opts["skin_type"] = metadata["skin_type_options"]
        if "tattoo_color_options" in metadata:
            opts["tattoo_color"] = metadata["tattoo_color_options"]
            opts["colors"] = metadata["tattoo_color_options"]
        if "tattoo_type_options" in metadata:
            opts["tattoo_type"] = metadata["tattoo_type_options"]
        if "laser_type_options" in metadata:
            opts["laser_type"] = metadata["laser_type_options"]
            opts["laser_type_sanitized"] = metadata["laser_type_options"]
    return opts

OPTIONS = extract_options(metadata)

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
    if not patient_id:
        return None
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
            row.get("colors") or row.get("tattoo_color"),
            float(row.get("size_cm2", 0) or 0),
            float(row.get("tattoo_age_years", 0) or 0),
            row.get("tattoo_type"),
            row.get("laser_type") or row.get("laser_type_sanitized"),
            float(row.get("repetition_rate_hz", 0) or 0),
            float(clinical_wavelength_by_color(row.get("colors") or row.get("tattoo_color"), row.get("laser_type") or row.get("laser_type_sanitized"), preds)),
            float(get_energy_value(preds)),
            float(get_pulse_duration_value(preds)),
            float(get_total_pulses_value(preds)),
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
        "history": "Patient History / Sessions",
        "new_prediction": "New Prediction",
        "about": "About",
        "patient_profile": "Patient Profile",
        "tattoo_characteristics": "Tattoo Characteristics",
        "device_inputs": "Device-Aware Inputs",
        "patient_information": "Patient Information",
        "treatment_sessions": "Treatment Sessions",
        "all_sessions": "All Saved Treatment Sessions",
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
        "history": "سجل المرضى والجلسات",
        "new_prediction": "تنبؤ جديد",
        "about": "حول النظام",
        "patient_profile": "ملف المريض",
        "tattoo_characteristics": "خصائص الوشم",
        "device_inputs": "مدخلات الجهاز",
        "patient_information": "معلومات المريض",
        "treatment_sessions": "جلسات العلاج",
        "all_sessions": "كل جلسات العلاج المحفوظة",
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
.block-container {
    padding-top: 4.8rem !important;
    padding-left: 3.1rem !important;
    padding-right: 3.1rem !important;
    max-width: 1180px;
}
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
# Session state
# ============================================================
if "language" not in st.session_state:
    st.session_state.language = "English"
if "view" not in st.session_state:
    st.session_state.view = "dashboard"
if "selected_patient_id" not in st.session_state:
    st.session_state.selected_patient_id = None
if "last_input" not in st.session_state:
    st.session_state.last_input = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "prediction_errors" not in st.session_state:
    st.session_state.prediction_errors = {}

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
    row = {}
    for f in FEATURES:
        if f == "skin_type":
            row[f] = skin_type
        elif f in ["colors", "tattoo_color"]:
            row[f] = colors
        elif f == "size_cm2":
            row[f] = float(size_cm2)
        elif f == "tattoo_age_years":
            row[f] = float(tattoo_age_years)
        elif f == "tattoo_type":
            row[f] = tattoo_type
        elif f in ["laser_type", "laser_type_sanitized"]:
            row[f] = laser_type
        elif f == "repetition_rate_hz":
            row[f] = float(repetition_rate_hz)
        else:
            row[f] = np.nan

    # Store friendly versions too for database/export even if model did not use both names
    row.setdefault("colors", colors)
    row.setdefault("tattoo_color", colors)
    row.setdefault("laser_type", laser_type)
    row.setdefault("laser_type_sanitized", laser_type)
    return row

def get_model_feature_names(model):
    """Return feature names expected by this specific trained model/pipeline."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return FEATURES

def value_for_feature(row, feature_name):
    """Map app inputs to the exact feature names used during training."""
    f = str(feature_name)
    key = f.lower().strip()

    # Skin type alternatives
    if key in ["skin_type", "skin type", "fitzpatrick", "fitzpatrick_scale"]:
        return row.get("skin_type")

    # Color alternatives
    if key in ["color", "colors", "tattoo_color", "tattoo color", "ink_color", "ink color"]:
        return row.get("tattoo_color", row.get("colors"))

    # Size alternatives
    if key in ["size", "tattoo_size", "tattoo size", "size_cm2", "tattoo_size_cm2", "tattoo area", "area"]:
        return float(row.get("size_cm2", 0))

    # Age alternatives
    if key in ["age", "tattoo_age", "tattoo age", "tattoo_age_years", "tattoo age years"]:
        return float(row.get("tattoo_age_years", 0))

    # Tattoo type alternatives
    if key in ["type", "tattoo_type", "tattoo type"]:
        return row.get("tattoo_type")

    # Laser type alternatives
    if key in ["laser_type", "laser type", "laser_type_sanitized", "device", "device_type", "device type"]:
        return row.get("laser_type_sanitized", row.get("laser_type"))

    # Repetition rate alternatives
    if key in ["repetition_rate", "repetition rate", "repetition_rate_hz", "repetition rate hz", "hz"]:
        return float(row.get("repetition_rate_hz", 0))

    # Exact match if present
    if f in row:
        return row[f]

    return np.nan

def predict_outputs(row):
    preds = {}
    errors = {}

    for target, model in MODELS.items():
        target_name = normalize_target_name(target)
        try:
            model_features = get_model_feature_names(model)
            model_row = {feature: value_for_feature(row, feature) for feature in model_features}
            X = pd.DataFrame([model_row], columns=model_features)

            value = model.predict(X)[0]
            if isinstance(value, (list, tuple, np.ndarray)):
                value = np.asarray(value).ravel()[0]
            preds[target_name] = float(value)
        except Exception as exc:
            preds[target_name] = np.nan
            errors[target_name] = str(exc)

    # Save diagnostic information so the UI can show it only when needed.
    st.session_state.prediction_errors = errors
    return preds

def get_prediction_value(preds, possible_names):
    """Read a prediction even if target names use spaces, underscores, or different case."""
    normalized = {normalize_target_name(k): v for k, v in preds.items()}
    for name in possible_names:
        canonical = normalize_target_name(name)
        if canonical in normalized:
            try:
                value = float(normalized[canonical])
                if np.isfinite(value):
                    return value
            except Exception:
                pass
    return np.nan



def get_ml_wavelength(preds):
    return get_prediction_value(
        preds,
        [
            "Wavelength_nm", "Wavelength", "wavelength_nm", "wavelength",
            "WAVELENGTH_NM", "WAVELENGTH"
        ],
    )


def get_energy_value(preds):
    return get_prediction_value(
        preds,
        [
            "Energy_Jcm2", "Energy", "energy_jcm2", "energy",
            "ENERGY_JCM2", "ENERGY"
        ],
    )


def get_pulse_duration_value(preds):
    return get_prediction_value(
        preds,
        [
            "Pulse Duration", "Pulse_Duration", "PulseDuration",
            "pulse_duration", "pulse duration", "PULSE_DURATION"
        ],
    )


def get_total_pulses_value(preds):
    return get_prediction_value(
        preds,
        [
            "Total Pulses", "Total_Pulses", "TotalPulses",
            "total_pulses", "total pulses", "TOTAL_PULSES"
        ],
    )


def clinical_wavelength_by_color(color, laser_type=None, preds=None):
    """Clinical-rule wavelength layer.

    Wavelength is constrained by tattoo ink color and available laser technology,
    so the interface uses a guideline rule for Wavelength and keeps ML models
    for Energy, Pulse Duration, and Total Pulses.
    """
    c = str(color or "").lower().strip()
    lt = str(laser_type or "").lower().strip()

    if c in ["black", "dark", "dark black", "grey", "gray"]:
        return 1064.0

    if c in ["red", "orange", "brown"]:
        return 532.0

    if c in ["green"]:
        if "ruby" in lt:
            return 694.0
        if "alex" in lt:
            return 755.0
        return 755.0

    if c in ["blue"]:
        if "ruby" in lt:
            return 694.0
        if "alex" in lt:
            return 755.0
        return 1064.0

    if c in ["purple", "violet", "yellow", "white"]:
        return 532.0

    if preds is not None:
        ml_value = get_ml_wavelength(preds)
        if np.isfinite(ml_value):
            return ml_value

    return np.nan

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
    for k, v in row.items():
        rows.append([k, v])
    final_wavelength = clinical_wavelength_by_color(
        row.get("colors") or row.get("tattoo_color"),
        row.get("laser_type") or row.get("laser_type_sanitized"),
        preds,
    )
    rows.append(["Final Wavelength_nm (Clinical Rule)", final_wavelength])
    rows.append(["Predicted Energy_Jcm2 (ML)", get_energy_value(preds)])
    rows.append(["Predicted Pulse Duration (ML)", get_pulse_duration_value(preds)])
    rows.append(["Predicted Total Pulses (ML)", get_total_pulses_value(preds)])
    rows.append(["Clinical Disclaimer", T["disclaimer"]])
    df = pd.DataFrame(rows, columns=["Field", "Value"])
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Prediction_Report", index=False)
    return out.getvalue()

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-logo"><div class="logo-icon">⚡</div><div class="logo-text">Laser DSS</div></div>', unsafe_allow_html=True)
    T_side = TEXT.get(st.session_state.get("language", "English"), TEXT["English"])

    if st.button("▦  " + T_side["dashboard"], use_container_width=True):
        st.session_state.view = "dashboard"
    if st.button("👥  " + T_side["patients"], use_container_width=True):
        st.session_state.view = "patients"
    if st.button("📋  " + T_side["history"], use_container_width=True):
        st.session_state.view = "history"
    if st.button("⚡  " + T_side["new_prediction"], use_container_width=True):
        st.session_state.view = "prediction"
        st.session_state.selected_patient_id = None
    if st.button("ℹ️  " + T_side["about"], use_container_width=True):
        st.session_state.view = "about"

    st.markdown("---")
    st.session_state.language = st.radio("🌐", ["English", "العربية"], index=0 if st.session_state.language == "English" else 1)
    st.markdown('<div class="user-box"><div class="user-avatar">S</div><div><b>Sury Hop</b><br><span class="muted">suryhop@gmail.com</span></div></div>', unsafe_allow_html=True)

T = TEXT[st.session_state.language]

# ============================================================
# Header
# ============================================================
st.markdown(
    f'<div class="page-title">{T["title"]}</div>'
    f'<div class="page-subtitle">{T["subtitle"]}</div>'
    '<span class="chip">Device-Aware ML</span>'
    '<span class="chip-active">✓ Guideline Support</span>'
    '<span class="chip">Clinical Review</span>',
    unsafe_allow_html=True,
)

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
            if not full_name.strip():
                st.warning(T["enter_patient_name"])
            else:
                add_patient(full_name, file_number, age, gender, notes)
                st.success(T["patient_saved"])
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    search = st.text_input("🔎 " + T["search"])
    view_df = patients_df.copy()
    if search.strip() and not view_df.empty:
        q = search.lower()
        view_df = view_df[
            view_df.full_name.str.lower().str.contains(q, na=False)
            | view_df.file_number.astype(str).str.lower().str.contains(q, na=False)
        ]
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

elif st.session_state.view == "history":
    st.markdown(f"<h1 style='color:#073b44'>{T['history']}</h1>", unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="section-title">📋 {T["all_sessions"]}</div>', unsafe_allow_html=True)
    if sessions_df.empty:
        st.info(T["no_sessions"])
    else:
        show_cols = [
            "created_at", "full_name", "file_number", "session_number", "skin_type",
            "colors", "size_cm2", "tattoo_age_years", "tattoo_type", "laser_type",
            "repetition_rate_hz", "wavelength", "energy", "pulse_duration", "total_pulses"
        ]
        existing_cols = [c for c in show_cols if c in sessions_df.columns]
        st.dataframe(sessions_df[existing_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download All Sessions CSV",
            sessions_df.to_csv(index=False).encode("utf-8-sig"),
            "all_treatment_sessions.csv",
            "text/csv",
            use_container_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.view == "patient_detail":
    patient_id = st.session_state.get("selected_patient_id")
    p = get_patient(patient_id) if patient_id else None
    if not p:
        st.warning("No patient selected.")
    else:
        if st.button("← " + T["back_to_patients"]):
            st.session_state.view = "patients"
            st.rerun()

        st.markdown(f'<div class="card"><div class="section-title">{T["patient_information"]}</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.write("**" + T["full_name"] + "**"); st.write(p["full_name"])
            st.write("**" + T["age"] + "**"); st.write(p["age"])
        with b:
            st.write("**" + T["file_number"] + "**"); st.write(p["file_number"])
            st.write("**" + T["gender"] + "**"); st.write(p["gender"])
        if p.get("notes"):
            st.write("**" + T["notes"] + "**")
            st.write(p["notes"])
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

    skin_options = safe_options(OPTIONS.get("skin_type", []), ["II", "III", "IV", "V", "VI"])
    color_options = clean_color_options(OPTIONS.get("colors", OPTIONS.get("tattoo_color", [])))
    tattoo_type_options = safe_options(OPTIONS.get("tattoo_type", []), ["simple", "complex", "amateur", "professional"])
    laser_type_options = safe_options(OPTIONS.get("laser_type", OPTIONS.get("laser_type_sanitized", [])), ["Q-switched Nd:YAG", "Picosecond", "Alexandrite", "Ruby"])

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(f'<div class="card"><div class="section-title">{T["patient_profile"]}</div>', unsafe_allow_html=True)
        skin_type = st.selectbox(T["skin_type"], skin_options)
        st.markdown(f'**{T["tattoo_age"]}**')
        y1, y2 = st.columns(2)
        with y1:
            tattoo_years = st.number_input(T["years"], 0, 80, 3, 1)
        with y2:
            tattoo_months = st.number_input(T["months"], 0, 11, 0, 1)
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
    with d1:
        laser_type = st.selectbox(T["laser_type"], laser_type_options)
    with d2:
        repetition_rate_hz = st.number_input(T["repetition_rate"], 0.0, 50.0, 5.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    row = build_row(skin_type, color, size_cm2, tattoo_age_years, tattoo_type, laser_type, repetition_rate_hz)

    if st.button("⚡ " + T["generate"], use_container_width=True):
        st.session_state.last_input = row
        st.session_state.last_prediction = predict_outputs(row)

    if st.session_state.last_prediction is not None:
        preds = st.session_state.last_prediction
        wavelength_value = clinical_wavelength_by_color(color, laser_type, preds)
        energy_value = get_energy_value(preds)
        pulse_duration_value = get_pulse_duration_value(preds)
        total_pulses_value = get_total_pulses_value(preds)

        w = int(round(wavelength_value)) if np.isfinite(wavelength_value) else "N/A"
        e = round(energy_value, 3) if np.isfinite(energy_value) else "N/A"
        pdur = round(pulse_duration_value, 4) if np.isfinite(pulse_duration_value) else "N/A"
        tp = int(round(max(total_pulses_value, 0))) if np.isfinite(total_pulses_value) else "N/A"

        # Temporary transparent diagnostic: appears only if any main output is missing.
        if "N/A" in [w, e, pdur, tp]:
            with st.expander("Technical note: available prediction keys", expanded=False):
                st.write(list(preds.keys()))
                st.write(preds)

        st.markdown(f'<div class="card"><div class="section-title">{T["recommended"]}</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            show_result_card("〰️", T["wavelength"], w, "nm · Clinical Rule")
        with r2:
            show_result_card("⚡", T["energy"], e, "J/cm² · ML")
        r3, r4 = st.columns(2)
        with r3:
            show_result_card("⏱️", T["pulse_duration"], pdur, "ns · ML")
        with r4:
            show_result_card("#", T["total_pulses"], tp, "pulses · ML")
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
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.view == "about":
    st.markdown(f'<div class="card"><div class="section-title">{T["about"]}</div>', unsafe_allow_html=True)
    st.write(T["about_text"])
    st.write(f"Model file loaded from: `{MODEL_PATH.name}`")
    st.write(f"Number of target outputs: {len(TARGETS)}")
    st.write("Target outputs:", ", ".join(TARGETS))
    st.write("Input features used by the model:")
    st.write(FEATURES)
    st.markdown(f'<div class="disclaimer">⚠️ <b>{T["clinical_disclaimer"]}:</b> {T["disclaimer"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
