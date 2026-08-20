import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import cv2
import base64
import io
import streamlit as st
import os
import zipfile

st.write("APP STARTED")

# ==========================================================
# CUSTOM LAYERS (harus didefinisikan sebelum load model,
# karena dipakai di dalam arsitektur CBAM)
# ==========================================================
@tf.keras.utils.register_keras_serializable()
class ChannelMean(layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)


@tf.keras.utils.register_keras_serializable()
class ChannelMax(layers.Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=-1, keepdims=True)


# ==========================================================
# KONFIGURASI
# ==========================================================
MODEL_ZIP = "best_model.zip"
MODEL_PATH = "best_model.keras"

IMG_SIZE = (224, 224)
THRESHOLD = 0.5
LAST_CONV_LAYER = "conv2d_22"
LABELS = {1: "Normal", 0: "Glaucoma"}

# Extract model jika belum ada
if not os.path.exists(MODEL_PATH):
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(".")

st.set_page_config(page_title="Glaucoma Screening", page_icon="\U0001F441", layout="wide")


# ==========================================================
# STYLE — tema klinis, warna diambil dari citra fundus itu sendiri
# ==========================================================
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root{
            --bg: #0A1516;
            --panel: #101F21;
            --panel-2: #0D1A1C;
            --border: #1E3437;
            --text: #EAF3F1;
            --muted: #7FA3A0;
            --glaucoma: #E8834B;
            --normal: #4FD1B0;
        }

        .stApp{
            background: radial-gradient(circle at 15% -10%, #12262A 0%, var(--bg) 45%) fixed;
        }

        html, body, [class*="css"]{
            font-family: 'Inter', sans-serif;
            color: var(--text);
        }

        .block-container{
            padding-top: 2.5rem;
            max-width: 1100px;
        }

        /* ---- Header ---- */
        .app-header{
            display:flex;
            align-items:center;
            gap:16px;
            margin-bottom: 4px;
        }
        .app-header .icon{
            font-size: 2.1rem;
            filter: drop-shadow(0 0 12px rgba(232,131,75,0.35));
        }
        .app-header h1{
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 2rem;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .app-subtitle{
            color: var(--muted);
            font-size: 0.95rem;
            margin: 6px 0 20px 0;
            padding-left: 2px;
        }
        .app-divider{
            height: 1px;
            background: linear-gradient(90deg, var(--glaucoma) 0%, var(--border) 18%, var(--border) 82%, var(--normal) 100%);
            opacity: 0.5;
            margin-bottom: 32px;
        }

        .eyebrow{
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 10px;
        }

        /* ---- File uploader ---- */
        [data-testid="stFileUploader"]{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 6px;
        }
        [data-testid="stFileUploaderDropzone"]{
            background: transparent;
        }

        /* ---- Equal-height card row ---- */
        div[data-testid="stHorizontalBlock"]{
            align-items: stretch;
        }
        div[data-testid="column"]{
            display: flex;
        }
        div[data-testid="column"] > div{
            width: 100%;
        }

        /* ---- Diagnosis card ---- */
        .diag-card{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 28px;
            display: flex;
            align-items: center;
            gap: 28px;
            height: 300px;
            box-sizing: border-box;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        }
        .status-chip{
            display:inline-flex;
            align-items:center;
            gap:8px;
            padding:7px 16px;
            border-radius:999px;
            font-family:'Space Grotesk', sans-serif;
            font-weight:600;
            font-size:1.05rem;
            border:1px solid var(--chip-color);
            color:var(--chip-color);
            background: rgba(232,131,75,0.08);
            margin-bottom: 14px;
        }
        .status-chip::before{
            content:"";
            width:8px;height:8px;border-radius:50%;
            background:var(--chip-color);
            box-shadow: 0 0 8px var(--chip-color);
        }
        .diag-note{
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.5;
            max-width: 320px;
        }

        /* ---- Radial gauge (signature element) ---- */
        .gauge{
            --pct-deg: calc(var(--pct) * 3.6deg);
            width: 148px; height: 148px;
            border-radius: 50%;
            flex-shrink: 0;
            background: conic-gradient(var(--chip-color) var(--pct-deg), #16292B var(--pct-deg));
            display:flex; align-items:center; justify-content:center;
            position: relative;
        }
        .gauge-inner{
            width: 112px; height:112px;
            border-radius:50%;
            background: var(--panel-2);
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            border: 1px solid var(--border);
        }
        .gauge-value{
            font-family:'IBM Plex Mono', monospace;
            font-size:1.55rem;
            font-weight:600;
            color: var(--text);
            line-height:1;
        }
        .gauge-label{
            font-size:0.62rem;
            letter-spacing:0.08em;
            text-transform:uppercase;
            color: var(--muted);
            margin-top:4px;
        }

        /* ---- Image frame ---- */
        .img-frame{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 10px;
            height: 300px;
            box-sizing: border-box;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            display: flex;
        }
        .img-frame img{
        border-radius:10px;
        width:100%;
        height:100%;
        object-fit:contain;
        background:black;
    }

        /* ---- Grad-CAM cards ---- */
        .gc-card{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.2);
        }
        .gc-card img{ border-radius: 8px; }
        .gc-title{
            font-family:'Space Grotesk', sans-serif;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--muted);
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin: 10px 2px 2px 2px;
        }

        .section-title{
            font-family:'Space Grotesk', sans-serif;
            font-size: 1.15rem;
            font-weight: 600;
            margin: 40px 0 16px 0;
        }

        .empty-state{
            border: 1px dashed var(--border);
            border-radius: 16px;
            padding: 48px;
            text-align: center;
            color: var(--muted);
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def img_to_base64(arr):
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


inject_css()


# ==========================================================
# LOAD MODEL (cached supaya gak reload tiap interaksi)
# ==========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"ChannelMean": ChannelMean, "ChannelMax": ChannelMax},
    )


model = load_model()


# ==========================================================
# GRAD-CAM (identik dengan versi Colab, Selvaraju et al. 2017)
# ==========================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return heatmap.numpy()
    heatmap = heatmap / max_val
    return heatmap.numpy()


def make_retina_mask(original_img, threshold=10):
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    mask = (gray > threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def overlay_gradcam(original_img, heatmap, retina_mask=None, alpha=0.4):

    h, w = original_img.shape[:2]

    heatmap = cv2.resize(heatmap, (w, h))

    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    if retina_mask is not None:

        mask = retina_mask.astype(np.uint8)

        heatmap_color = cv2.bitwise_and(
            heatmap_color,
            heatmap_color,
            mask=mask
        )

    overlay = cv2.addWeighted(
        original_img,
        1-alpha,
        heatmap_color,
        alpha,
        0
    )

    return overlay, heatmap_color


# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    """
    <div class="app-header">
        <div class="icon">&#128065;</div>
        <h1>Glaucoma Screening</h1>
    </div>
    <div class="app-subtitle">CNN + CBAM &middot; klasifikasi citra fundus retina dengan visualisasi Grad-CAM</div>
    <div class="app-divider"></div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Upload citra fundus</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    original_resized = np.array(pil_image.resize(IMG_SIZE))

    img_array = original_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    with st.spinner("Menganalisis citra..."):
        prob = float(model.predict(img_batch, verbose=0)[0][0])
        is_glaucoma = prob < THRESHOLD
        pred_label = LABELS[0] if is_glaucoma else LABELS[1]
        confidence = (1 - prob) if is_glaucoma else prob
        chip_color = "var(--glaucoma)" if is_glaucoma else "var(--normal)"

    gradcam_ok = True
    gradcam_error = None
    try:
        heatmap = make_gradcam_heatmap(img_batch, model, LAST_CONV_LAYER)
        mask = make_retina_mask(original_resized)
        overlay, heatmap_color = overlay_gradcam(
            original_resized,
            heatmap,
            retina_mask=mask,
            alpha=0.4
        )

    except Exception as e:
        gradcam_ok = False
        gradcam_error = str(e)

    st.markdown('<div class="section-title" style="margin-top:32px;">Hasil Analisis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown('<div class="eyebrow">Diagnosis</div>', unsafe_allow_html=True)
        note = (
            "Ditemukan pola yang konsisten dengan glaucoma pada citra ini."
            if is_glaucoma
            else "Tidak ditemukan indikasi glaucoma yang signifikan pada citra ini."
        )
        st.markdown(
            f"""
            <div class="diag-card">
                <div class="gauge" style="--pct:{confidence*100:.2f}; --chip-color:{chip_color};">
                    <div class="gauge-inner">
                        <span class="gauge-value">{confidence*100:.0f}%</span>
                        <span class="gauge-label">keyakinan</span>
                    </div>
                </div>
                <div>
                    <div class="status-chip" style="--chip-color:{chip_color};">{pred_label}</div>
                    <div class="diag-note">{note}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="eyebrow">Citra Input</div>', unsafe_allow_html=True)
        b64 = img_to_base64(original_resized)
        st.markdown(
            f'<div class="img-frame"><img src="data:image/png;base64,{b64}" style="width:100%;display:block;"/></div>',
            unsafe_allow_html=True,
        )

    if gradcam_ok:
        st.markdown('<div class="section-title">Visualisasi Grad-CAM</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, arr, title in [
            (c1, original_resized, "Original"),
            (c2, heatmap_color, "Heatmap"),
            (c3, overlay, "Overlay"),
        ]:
            with col:
                b64_im = img_to_base64(arr)
                st.markdown(
                    f'<div class="gc-card"><img src="data:image/png;base64,{b64_im}" style="width:100%;display:block;"/></div>'
                    f'<div class="gc-title">{title}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.warning(
            f"Grad-CAM gagal dijalankan: {gradcam_error}\n\n"
            f"Cek nama layer conv terakhir via `model.summary()` di Colab, "
            f"kemungkinan berbeda dari '{LAST_CONV_LAYER}'."
        )
else:
    st.markdown(
        '<div class="empty-state">Belum ada citra yang diunggah &mdash; upload gambar fundus retina untuk memulai analisis.</div>',
        unsafe_allow_html=True,
    )