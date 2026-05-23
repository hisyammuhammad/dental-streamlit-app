"""
🦷 Dental Disease Classifier — Streamlit Cloud Edition
Model: MobileNetV2 Transfer Learning
Source: Hugging Face Hub (tanpa MLflow, siap deploy ke share.streamlit.io)

Deploy steps:
1. Upload model ke HuggingFace:
       huggingface-cli upload <username>/dental-classifier models/ .
2. Push repo ini ke GitHub (app.py + requirements.txt + metadata.json)
3. Deploy di https://share.streamlit.io
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# KONFIGURASI — sesuaikan HF_REPO_ID dengan akun Hugging Face kamu
# ─────────────────────────────────────────────────────────────────
HF_REPO_ID    = "hisyamyahya/dental-classifier"   # ← GANTI dengan username/repo kamu
MODEL_FILENAME = "dental_mobilenetv2_final.keras"
META_FILENAME  = "metadata.json"
IMG_SIZE       = (224, 224)

# Metadata fallback jika file tidak ada di repo
FALLBACK_META = {
    "idx_to_class"  : {"0": "Caries", "1": "Gingivitas"},
    "img_size"      : [224, 224],
    "best_threshold": 0.5,
    "model_base"    : "MobileNetV2",
    "test_accuracy" : None,
    "test_auc"      : None,
}

# ─────────────────────────────────────────────────────────────────
# LOAD MODEL dari Hugging Face Hub
# @st.cache_resource → hanya load sekali, tidak reload tiap interaksi
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Mengunduh & memuat model dari Hugging Face...")
def load_model_and_meta():
    """
    Download model .keras dan metadata.json dari Hugging Face Hub,
    lalu load ke memory. Di-cache agar tidak re-download tiap refresh.

    Preprocessing WAJIB: preprocess_input MobileNetV2 (range [-1, 1])
    BUKAN rescale /255 — itu menyebabkan prediksi salah.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        st.error("Package `huggingface_hub` tidak terinstall. Tambahkan ke requirements.txt")
        st.stop()

    try:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as _pi
    except ImportError:
        st.error("Package `tensorflow-cpu` tidak terinstall. Tambahkan ke requirements.txt")
        st.stop()

    # Download model
    with st.spinner("Mengunduh model (~15MB)..."):
        try:
            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                repo_type="model",
            )
        except Exception as e:
            st.error(f"❌ Gagal download model dari Hugging Face: {e}")
            st.info(
                f"Pastikan:\n"
                f"1. `HF_REPO_ID = '{HF_REPO_ID}'` sudah benar\n"
                f"2. File `{MODEL_FILENAME}` sudah diupload ke repo\n"
                f"3. Repo bersifat **Public**"
            )
            st.stop()

    # Download metadata
    try:
        meta_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=META_FILENAME,
            repo_type="model",
        )
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        st.warning("⚠️ metadata.json tidak ditemukan di HF repo — menggunakan fallback.")
        meta = FALLBACK_META

    # Load model
    with st.spinner("Memuat model ke memori..."):
        model = tf.keras.models.load_model(
            model_path,
            compile=False,   # tidak perlu compile untuk inference
        )

    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    threshold    = float(meta.get("best_threshold", 0.5))

    return model, idx_to_class, threshold, meta


# ─────────────────────────────────────────────────────────────────
# PREPROCESSING — identik dengan pipeline training notebook
# ─────────────────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    PIL Image → numpy array siap prediksi.
    WAJIB gunakan preprocess_input MobileNetV2 (range [-1, 1]).
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img   = pil_image.convert("RGB").resize(IMG_SIZE)
    arr   = np.array(img, dtype=np.float32)
    arr   = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)


# ─────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────
def predict(model, img_array, idx_to_class, threshold):
    raw_prob   = float(model.predict(img_array, verbose=0)[0][0])
    pred_idx   = int(raw_prob >= threshold)
    label      = idx_to_class[pred_idx]
    confidence = raw_prob if pred_idx == 1 else (1.0 - raw_prob)

    if confidence >= 0.90:
        risk_label = "⚠️ Tinggi"
        risk_color = "#dc2626"
    elif confidence >= 0.70:
        risk_label = "🔶 Sedang"
        risk_color = "#d97706"
    else:
        risk_label = "🔵 Rendah"
        risk_color = "#2563eb"

    return {
        "label"      : label,
        "confidence" : confidence,
        "raw_prob"   : raw_prob,
        "pred_idx"   : pred_idx,
        "risk_label" : risk_label,
        "risk_color" : risk_color,
    }


# ─────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────
def render_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    /* ── Header ── */
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 4px;
        margin-bottom: 0;
    }

    /* ── Result card ── */
    .result-card {
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        margin: 0.8rem 0;
        border: 2px solid;
    }
    .result-card.caries {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        border-color: #fb923c;
    }
    .result-card.gingivitas {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-color: #4ade80;
    }
    .result-label {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 4px 0;
        color: #0f172a;
    }
    .result-meta {
        font-size: 0.88rem;
        color: #475569;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: #f8fafc;
        border-left: 3px solid #94a3b8;
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 1rem;
        line-height: 1.5;
    }

    /* ── Upload area ── */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0f172a;
    }
    section[data-testid="stSidebar"] h3 {
        color: #7dd3fc !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] strong {
        color: #f1f5f9 !important;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Dental Classifier 🦷",
        page_icon="🦷",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "Dental Disease Classifier — MobileNetV2 Transfer Learning\nDibuat untuk keperluan akademis."
        }
    )

    render_css()

    # ── Load model ────────────────────────────────────────────────
    model, idx_to_class, threshold, meta = load_model_and_meta()

    # ── SIDEBAR ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🦷 Dental Classifier")
        st.markdown("---")

        st.markdown("### ℹ️ Model")
        st.markdown(f"**Base:** `{meta.get('model_base', 'MobileNetV2')}`")
        st.markdown(f"**Input:** `{meta.get('img_size', [224,224])}` px")
        st.markdown("**Preprocessing:** `preprocess_input`")

        acc = meta.get("test_accuracy")
        auc = meta.get("test_auc")
        if acc is not None:
            st.markdown(f"**Test Accuracy:** `{acc:.4f}`")
        if auc is not None:
            st.markdown(f"**Test AUC:** `{auc:.4f}`")

        st.markdown("---")
        st.markdown("### 🏷️ Kelas")
        for idx, cls in sorted(idx_to_class.items()):
            icon = "🦷" if "aries" in cls else "🌿"
            st.markdown(f"- `{idx}` → {icon} **{cls}**")

        st.markdown("---")
        st.markdown("### 🎚️ Threshold")
        custom_threshold = st.slider(
            "Sesuaikan threshold prediksi",
            min_value=0.05,
            max_value=0.95,
            value=float(threshold),
            step=0.05,
            help=(
                "Threshold dipilih dari validation set via weighted F1.\n\n"
                "Turunkan → lebih sensitif (tangkap lebih banyak kasus).\n"
                "Naikkan → lebih spesifik (kurangi false positive)."
            )
        )

        st.markdown("---")
        st.caption("🔗 Model tersimpan di Hugging Face Hub")
        st.caption(f"`{HF_REPO_ID}`")
        st.caption("📚 Dibuat untuk keperluan akademis — Universitas Amikom Yogyakarta")

    # ── HEADER ────────────────────────────────────────────────────
    st.markdown('<p class="app-title">🦷 Dental Disease Classifier</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">'
        'MobileNetV2 · Transfer Learning · TensorFlow · Universitas Amikom Yogyakarta'
        '</p>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # ── TABS ──────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📸 Prediksi Gambar",
        "📂 Prediksi Batch",
        "📊 Info Model",
    ])

    # ═══════════════════════════════════════════════════════════════
    # TAB 1 — Single Image Prediction
    # ═══════════════════════════════════════════════════════════════
    with tab1:
        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            st.subheader("Upload Gambar Gigi")
            uploaded = st.file_uploader(
                "Pilih gambar foto gigi",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="single",
                help="Foto gigi dari kamera/klinik dalam format JPG atau PNG"
            )

            if uploaded:
                pil_img = Image.open(uploaded)
                st.image(pil_img, caption=f"📎 {uploaded.name}", use_container_width=True)
            else:
                # Placeholder saat belum upload
                st.markdown("""
                <div style="
                    border: 2px dashed #cbd5e1;
                    border-radius: 12px;
                    padding: 3rem 2rem;
                    text-align: center;
                    color: #94a3b8;
                ">
                    <p style="font-size:2rem;margin:0">📷</p>
                    <p style="margin:0.5rem 0 0">Upload foto gigi di atas</p>
                </div>
                """, unsafe_allow_html=True)

        with col_result:
            st.subheader("Hasil Prediksi")

            if not uploaded:
                st.markdown("""
                **Cara penggunaan:**
                1. Upload foto gigi (JPG/PNG) di sebelah kiri
                2. Tunggu analisis (~1–3 detik)
                3. Lihat prediksi dan confidence score
                4. Sesuaikan threshold di sidebar jika perlu

                ---
                **Kelas yang dideteksi:**
                - 🦷 **Caries** — kerusakan gigi akibat bakteri
                - 🌿 **Gingivitas** — peradangan pada gusi
                """)
            else:
                with st.spinner("🔍 Menganalisis gambar..."):
                    try:
                        arr    = preprocess_image(pil_img)
                        result = predict(model, arr, idx_to_class, custom_threshold)
                    except Exception as e:
                        st.error(f"❌ Error saat prediksi: {e}")
                        st.stop()

                label      = result["label"]
                confidence = result["confidence"]
                raw_prob   = result["raw_prob"]
                risk_label = result["risk_label"]
                risk_color = result["risk_color"]

                # Card utama
                is_caries   = "aries" in label.lower()
                card_class  = "caries" if is_caries else "gingivitas"
                icon        = "🦷" if is_caries else "🌿"

                st.markdown(f"""
                <div class="result-card {card_class}">
                    <p class="result-label">{icon} {label}</p>
                    <p class="result-meta">
                        Confidence: <strong>{confidence:.1%}</strong>
                        &nbsp;·&nbsp;
                        Risk: <strong style="color:{risk_color}">{risk_label}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence", f"{confidence:.1%}")
                m2.metric("Raw Prob",   f"{raw_prob:.4f}")
                m3.metric("Threshold",  f"{custom_threshold:.2f}")

                # Probability bar per kelas
                st.markdown("**Distribusi Probabilitas**")
                for idx_cls, cls_name in sorted(idx_to_class.items()):
                    p    = raw_prob if idx_cls == 1 else (1.0 - raw_prob)
                    icon = "🦷" if "aries" in cls_name else "🌿"
                    st.progress(float(p), text=f"{icon} {cls_name}: {p:.1%}")

                # Expander detail teknis
                with st.expander("🔧 Detail Teknis"):
                    st.json({
                        "label"         : label,
                        "confidence"    : round(confidence, 4),
                        "raw_probability": round(raw_prob, 4),
                        "threshold_used": custom_threshold,
                        "pred_class_idx": result["pred_idx"],
                        "preprocessing" : "mobilenet_v2.preprocess_input [-1, 1]",
                        "model"         : meta.get("model_base", "MobileNetV2"),
                        "hf_repo"       : HF_REPO_ID,
                    })

                # Disclaimer medis
                st.markdown("""
                <div class="disclaimer">
                    ⚕️ <strong>Disclaimer Medis:</strong>
                    Hasil klasifikasi ini bersifat <strong>informatif</strong> dan
                    <strong>tidak menggantikan diagnosis dokter gigi profesional</strong>.
                    Selalu konsultasikan kondisi gigi Anda dengan tenaga medis yang kompeten.
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # TAB 2 — Batch Prediction
    # ═══════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Prediksi Banyak Gambar Sekaligus")
        st.caption("Upload beberapa gambar, hasilnya bisa didownload sebagai CSV.")

        batch_files = st.file_uploader(
            "Upload gambar (bisa pilih banyak sekaligus)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            key="batch",
        )

        if batch_files:
            st.info(f"**{len(batch_files)} gambar** siap diproses dengan threshold `{custom_threshold:.2f}`")

            if st.button("🚀 Jalankan Prediksi Batch", type="primary", use_container_width=True):
                results_list = []
                progress_bar = st.progress(0, text="Memulai...")
                status_area  = st.empty()

                for i, f in enumerate(batch_files):
                    status_area.text(f"⏳ Memproses: {f.name}  ({i+1}/{len(batch_files)})")
                    try:
                        img       = Image.open(f).convert("RGB")
                        arr       = preprocess_image(img)
                        res       = predict(model, arr, idx_to_class, custom_threshold)
                        results_list.append({
                            "No"        : i + 1,
                            "Nama File" : f.name,
                            "Prediksi"  : res["label"],
                            "Confidence": f"{res['confidence']:.2%}",
                            "Raw Prob"  : round(res["raw_prob"], 4),
                            "Risk"      : res["risk_label"],
                            "Status"    : "✅ OK",
                        })
                    except Exception as e:
                        results_list.append({
                            "No"        : i + 1,
                            "Nama File" : f.name,
                            "Prediksi"  : "ERROR",
                            "Confidence": "-",
                            "Raw Prob"  : -1,
                            "Risk"      : "-",
                            "Status"    : f"❌ {str(e)[:50]}",
                        })

                    progress_bar.progress(
                        (i + 1) / len(batch_files),
                        text=f"Memproses {i+1}/{len(batch_files)}..."
                    )

                status_area.success("✅ Selesai!")

                df = pd.DataFrame(results_list)

                # Ringkasan
                valid_df = df[df["Prediksi"] != "ERROR"]
                st.markdown("#### Ringkasan")
                cols = st.columns(2 + len(idx_to_class))
                cols[0].metric("Total Gambar", len(df))
                cols[1].metric("Berhasil", len(valid_df))
                for j, (_, cls_name) in enumerate(sorted(idx_to_class.items())):
                    count = len(valid_df[valid_df["Prediksi"] == cls_name])
                    cols[2 + j].metric(cls_name, count)

                st.markdown("#### Hasil Detail")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Download
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Hasil (.csv)",
                    data=csv,
                    file_name="dental_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        else:
            st.markdown("""
            **Cara pakai:**
            1. Klik tombol upload di atas
            2. Pilih banyak file sekaligus (Ctrl+klik / Cmd+klik)
            3. Klik **Jalankan Prediksi Batch**
            4. Download hasilnya sebagai CSV
            """)

    # ═══════════════════════════════════════════════════════════════
    # TAB 3 — Model Info
    # ═══════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Informasi Model")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Arsitektur")
            st.code("""
MobileNetV2 (pretrained ImageNet)
  ↓ frozen pada Fase 1 training
GlobalAveragePooling2D
Dense(128)              ← tanpa aktivasi dulu
BatchNormalization
Activation('relu')
Dropout(0.4)
Dense(1, sigmoid)       ← binary output
            """, language="text")

            st.markdown("#### Pipeline Training")
            st.markdown("""
| Fase | Detail |
|------|--------|
| **Fase 1** | Feature Extraction — base frozen, LR=1e-3 |
| **Fase 2** | Fine-tuning — 30 layer unfrozen, LR=1e-5 |
| **Augmentasi** | flip, brightness, contrast, rot90 |
| **Class weight** | balanced (atasi imbalance) |
| **Threshold** | dicari dari val set via weighted F1 |
| **Preprocessing** | `preprocess_input` MobileNetV2 (range [-1, 1]) |
            """)

            st.markdown("#### Dataset")
            st.markdown(f"""
- **Caries** : {meta.get('dataset_caries', '~800')} gambar
- **Gingivitas** : {meta.get('dataset_gingivitas', '~732')} gambar
- **Split** : 70% train · 15% val · 15% test
            """)

        with col_b:
            st.markdown("#### Metadata Model")
            display_meta = {k: v for k, v in meta.items()
                           if k not in ("class_map",)}
            st.json(display_meta)

            st.markdown("#### Performa")
            acc = meta.get("test_accuracy")
            auc = meta.get("test_auc")
            if acc or auc:
                pm1, pm2 = st.columns(2)
                if acc:
                    pm1.metric("Test Accuracy", f"{acc:.4f}", f"{acc*100:.2f}%")
                if auc:
                    pm2.metric("Test AUC", f"{auc:.4f}")
            else:
                st.caption("Metrik belum tersedia di metadata.")

            st.markdown("#### Sumber Model")
            st.markdown(f"""
- 🤗 **Hugging Face:** `{HF_REPO_ID}`
- 📁 **File:** `{MODEL_FILENAME}`
- 🔧 **Framework:** TensorFlow / Keras
            """)

        st.markdown("---")
        st.warning("""
**⚠️ Catatan Penting — val_accuracy = 1.0:**

Model menunjukkan val_accuracy sempurna sejak epoch awal.
Ini karena dataset kecil (~1500 gambar) dan MobileNetV2 sangat powerful untuk tugas ini.
Sebelum production, disarankan uji dengan gambar dari sumber/klinik yang berbeda
untuk memvalidasi kemampuan generalisasi model.
        """)

    # ── FOOTER ────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "🦷 Dental Disease Classifier · "
        "MobileNetV2 Transfer Learning · "
        "Universitas Amikom Yogyakarta · "
        "Dibuat untuk keperluan akademis"
    )


if __name__ == "__main__":
    main()
