import streamlit as st
import numpy as np
import tensorflow as tf

# ============================================================
# ⚙️ CONFIG — แก้ค่าตรงนี้ให้ตรงกับวิธีเทรนโมเดลจริงของคุณ
# ============================================================
MODEL_PATH = "041titanic_mlp.keras"
SYSTEM_TITLE = "ระบบทำนายการรอดชีวิตผู้โดยสารเรือไททานิค"
DEVELOPER_NAME = "นายสถาพร ขวาธิจักร"

# ค่าความแม่นยำของโมเดล (ใส่ค่าจริงจากตอนเทรน/ประเมินผล)
MODEL_ACCURACY = 0.82  # <-- แก้เป็นค่าจริง เช่น 0.8212

# ลำดับฟีเจอร์ที่โมเดลถูกเทรนมา (สมมติฐาน: ไม่ได้ผ่านการ scale)
# ถ้าโมเดลจริงเทรนด้วยข้อมูลที่ scale แล้ว (StandardScaler/MinMaxScaler)
# ให้ใส่สูตรแปลงค่าใน apply_scaling() ด้านล่าง
FEATURE_NAMES = ["Pclass", "Sex", "Age", "SibSp", "Fare"]


def apply_scaling(features: np.ndarray) -> np.ndarray:
    """
    ปรับค่าฟีเจอร์ก่อนป้อนเข้าโมเดล
    ค่าเริ่มต้น: ไม่ scale (ใช้ค่าดิบตรง ๆ)
    ถ้าโมเดลเทรนด้วยข้อมูลที่ normalize ไว้ ให้แก้ฟังก์ชันนี้
    เช่น (features - mean) / std  หรือ  (features - min) / (max - min)
    """
    return features


# ============================================================
# 🎨 PAGE SETUP
# ============================================================
st.set_page_config(
    page_title=SYSTEM_TITLE,
    page_icon="🚢",
    layout="centered",
)

st.markdown(
    """
    <style>
        .main { background-color: #FAFAFA; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 700px; }
        h1 { color: #1F2A44; font-weight: 700; }
        .stButton>button {
            background-color: #1F2A44;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            border: none;
            width: 100%;
            font-weight: 600;
        }
        .stButton>button:hover { background-color: #35406B; }
        .result-box {
            padding: 1.2rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 1rem;
        }
        .survived { background-color: #E6F4EA; color: #1E7B34; }
        .not-survived { background-color: #FDECEC; color: #B3261E; }
        .accuracy-badge {
            display: inline-block;
            background-color: #EEF1F7;
            color: #1F2A44;
            padding: 0.3rem 0.9rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .footer {
            text-align: center;
            color: #8A8F98;
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #E5E7EB;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 📦 LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    load_error = str(e)

# ============================================================
# 🖥️ HEADER
# ============================================================
st.markdown(f"<h1 style='text-align:center;'>🚢 {SYSTEM_TITLE}</h1>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center; margin-bottom:1.2rem;'>"
    f"<span class='accuracy-badge'>ความแม่นยำของโมเดล: {MODEL_ACCURACY*100:.2f}%</span>"
    f"</div>",
    unsafe_allow_html=True,
)
st.write("กรอกข้อมูลผู้โดยสารด้านล่าง เพื่อทำนายโอกาสการรอดชีวิต")

if not model_loaded:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {load_error}")
    st.stop()

st.divider()

# ============================================================
# 📝 INPUT FORM
# ============================================================
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("ชั้นโดยสาร (Pclass)", options=[1, 2, 3], index=2)
    sex = st.radio("เพศ", options=["ชาย", "หญิง"], horizontal=True)
    age = st.slider("อายุ (Age)", min_value=0, max_value=90, value=28)

with col2:
    sibsp = st.number_input("จำนวนพี่น้อง/คู่สมรสที่ร่วมเดินทาง (SibSp)", min_value=0, max_value=10, value=0)
    fare = st.number_input("ค่าโดยสาร (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

sex_encoded = 0 if sex == "ชาย" else 1

st.write("")
predict_clicked = st.button("🔮 ทำนายผล")

# ============================================================
# 🔮 PREDICTION
# ============================================================
if predict_clicked:
    features = np.array([[pclass, sex_encoded, age, sibsp, fare]], dtype=np.float32)
    features = apply_scaling(features)

    prob = float(model.predict(features, verbose=0)[0][0])
    survived = prob >= 0.5

    if survived:
        st.markdown(
            f"<div class='result-box survived'>✅ คาดว่า <b>รอดชีวิต</b><br>"
            f"ความน่าจะเป็น: {prob*100:.1f}%</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='result-box not-survived'>❌ คาดว่า <b>ไม่รอดชีวิต</b><br>"
            f"ความน่าจะเป็น: {(1-prob)*100:.1f}%</div>",
            unsafe_allow_html=True,
        )

    with st.expander("รายละเอียดข้อมูลที่ใช้ทำนาย"):
        st.json(dict(zip(FEATURE_NAMES, [pclass, sex_encoded, age, sibsp, fare])))

# ============================================================
# 👤 FOOTER
# ============================================================
st.markdown(
    f"<div class='footer'>พัฒนาโดย {DEVELOPER_NAME}</div>",
    unsafe_allow_html=True,
)
