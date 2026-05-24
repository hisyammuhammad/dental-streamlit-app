"""
🦷 Dental Disease Classifier — Streamlit Cloud Edition
Strategi: Model disimpan di GitHub repo (dengan Git LFS)
Deploy target: https://share.streamlit.io

Struktur repo GitHub:
    dental-streamlit-app/
    ├── app.py
    ├── requirements.txt
    ├── metadata.json
    └── models/
        └── dental_mobilenetv2_final.keras  (via Git LFS)
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
# ═════════════════════════════════════════════════════════════════
# KONFIGURASI
# ═════════════════════════════════════════════════════════════════

# Path relatif — bekerja di lokal maupun Streamlit Cloud
BASE_DIR      = Path(__file__).resolve().parent
MODEL_PATH    = BASE_DIR / "models" / "dental_mobilenetv2_final.keras"
METADATA_PATH = BASE_DIR / "metadata.json"
IMG_SIZE      = (224, 224)

# Fallback metadata jika file belum ada
FALLBACK_META = {
    "idx_to_class"  : {"0": "Caries", "1": "Gingivitas"},
    "img_size"      : [224, 224],
    "best_threshold": 0.5,
    "model_base"    : "MobileNetV2",
    "test_accuracy" : None,
    "test_auc"      : None,
}


# ═════════════════════════════════════════════════════════════════
# LOAD MODEL
# @st.cache_resource → hanya load sekali per session
# ═════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⏳ Memuat model MobileNetV2...")
def load_model_and_meta():
    """
    Load model .keras dan metadata.json dari folder lokal repo.
    Preprocessing WAJIB: preprocess_input MobileNetV2 (range [-1,1]).
    BUKAN /255 — itu menyebabkan prediksi salah.
    """
    import tensorflow as tf

    # ── Load metadata ─────────────────────────────────────────────
    if METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        st.warning("⚠️ metadata.json tidak ditemukan, menggunakan default.")
        meta = FALLBACK_META

    # ── Load model ────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        st.error(
            f"❌ Model tidak ditemukan di `{MODEL_PATH}`\n\n"
            "Pastikan file `dental_mobilenetv2_final.keras` sudah ada "
            "di folder `models/` dan sudah di-push ke GitHub dengan Git LFS.\n\n"
            "Jalankan di terminal:\n"
            "```\ngit lfs track 'models/*.keras'\ngit add .\ngit commit -m 'add model'\ngit push\n```"
        )
        st.stop()

    model = keras.models.load_model("models/dental_mobilenetv2_final.keras")

    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    threshold    = float(meta.get("best_threshold", 0.5))
    return model, idx_to_class, threshold, meta


# ═════════════════════════════════════════════════════════════════
# PREPROCESSING — identik dengan training notebook
# ═════════════════════════════════════════════════════════════════

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    PIL → numpy → preprocess_input MobileNetV2 → batch dimension.
    Range output: [-1, 1] — WAJIB sama dengan training.
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


# ═════════════════════════════════════════════════════════════════
# PREDICT
# ═════════════════════════════════════════════════════════════════

def run_predict(model, img_array, idx_to_class, threshold):
    raw_prob   = float(model.predict(img_array, verbose=0)[0][0])
    pred_idx   = int(raw_prob >= threshold)
    label      = idx_to_class[pred_idx]
    confidence = raw_prob if pred_idx == 1 else (1.0 - raw_prob)

    if confidence >= 0.90:
        risk_text  = "Tinggi"
        risk_color = "#dc2626"
        risk_icon  = "🔴"
    elif confidence >= 0.70:
        risk_text  = "Sedang"
        risk_color = "#d97706"
        risk_icon  = "🟠"
    else:
        risk_text  = "Rendah"
        risk_color = "#2563eb"
        risk_icon  = "🔵"

    return {
        "label"      : label,
        "confidence" : confidence,
        "raw_prob"   : raw_prob,
        "pred_idx"   : pred_idx,
        "risk_text"  : risk_text,
        "risk_color" : risk_color,
        "risk_icon"  : risk_icon,
    }


# ═════════════════════════════════════════════════════════════════
# CSS
# ═════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    /* ── Header ─────────────────────────────── */
    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 0 0 4px 0;
    }
    .app-sub {
        font-size: 0.82rem;
        color: #64748b;
        font-family: 'IBM Plex Mono', monospace;
        margin: 0 0 1.5rem 0;
    }

    /* ── Result card ─────────────────────────── */
    .result-card {
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin: 0.8rem 0 1rem;
        border: 2px solid;
    }
    .result-card.caries {
        background: linear-gradient(135deg, #fff7ed, #ffedd5);
        border-color: #fb923c;
    }
    .result-card.gingivitas {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-color: #4ade80;
    }
    .result-label {
        font-size: 1.55rem;
        font-weight: 700;
        margin: 0 0 6px;
        color: #0f172a;
    }
    .result-sub {
        font-size: 0.85rem;
        color: #475569;
        font-family: 'IBM Plex Mono', monospace;
        margin: 0;
    }

    /* ── Upload placeholder ───────────────────── */
    .upload-placeholder {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 3rem 1rem;
        text-align: center;
        color: #94a3b8;
    }

    /* ── Disclaimer ──────────────────────────── */
    .disclaimer {
        background: #f8fafc;
        border-left: 3px solid #94a3b8;
        padding: 0.65rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.77rem;
        color: #64748b;
        margin-top: 1.2rem;
        line-height: 1.55;
    }

    /* ── Sidebar ─────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0f172a !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #7dd3fc !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] li {
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] strong {
        color: #f1f5f9 !important;
    }

    /* ── Metric cards ────────────────────────── */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Dental Classifier 🦷",
        page_icon="🦷",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                "🦷 Dental Disease Classifier\n"
                "Model: MobileNetV2 Transfer Learning\n"
                "Dibuat untuk keperluan akademis — Universitas Amikom Yogyakarta"
            )
        }
    )

    inject_css()

    # ── Load model ─────────────────────────────────────────────────
    model, idx_to_class, threshold, meta = load_model_and_meta()

    # ── SIDEBAR ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🦷 Dental Classifier")
        st.markdown("---")

        st.markdown("### ℹ️ Model")
        st.markdown(f"**Base:** `{meta.get('model_base','MobileNetV2')}`")
        st.markdown(f"**Input:** `224 × 224` px")
        st.markdown("**Preprocessing:** `preprocess_input` [-1, 1]")

        acc = meta.get("test_accuracy")
        auc = meta.get("test_auc")
        if acc is not None:
            st.markdown(f"**Test Accuracy:** `{acc:.4f}`")
        if auc is not None:
            st.markdown(f"**Test AUC:** `{auc:.4f}`")

        st.markdown("---")
        st.markdown("### 🏷️ Kelas")
        for i, cls in sorted(idx_to_class.items()):
            icon = "🦷" if "aries" in cls else "🌿"
            st.markdown(f"- `{i}` → {icon} **{cls}**")

        st.markdown("---")
        st.markdown("### 🎚️ Threshold")
        custom_threshold = st.slider(
            "Sesuaikan threshold prediksi",
            min_value=0.05,
            max_value=0.95,
            value=float(threshold),
            step=0.05,
            help=(
                "Default dipilih dari validation set.\n\n"
                "Turunkan → lebih sensitif (tangkap lebih banyak kasus).\n"
                "Naikkan → lebih spesifik (kurangi false positive)."
            )
        )

        st.markdown("---")
        st.caption("📚 Universitas Amikom Yogyakarta")
        st.caption("Dibuat untuk keperluan akademis")

    # ── HEADER ─────────────────────────────────────────────────────
    st.markdown('<p class="app-title">🦷 Dental Disease Classifier</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-sub">MobileNetV2 · Transfer Learning · TensorFlow · '
        'Universitas Amikom Yogyakarta</p>',
        unsafe_allow_html=True
    )

    # ── TABS ───────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📸 Prediksi Gambar",
        "📂 Prediksi Batch",
        "📊 Info Model",
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — Single Image Prediction
    # ══════════════════════════════════════════════════════════════
    with tab1:
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.subheader("Upload Gambar Gigi")
            uploaded = st.file_uploader(
                "Pilih foto gigi (JPG / PNG / JPEG)",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="single",
            )
            if uploaded:
                pil_img = Image.open(uploaded)
                st.image(pil_img, caption=f"📎 {uploaded.name}", use_container_width=True)
            else:
                st.markdown("""
                <div class="upload-placeholder">
                    <p style="font-size:2.5rem;margin:0">📷</p>
                    <p style="margin:0.5rem 0 0 0">Upload foto gigi di atas</p>
                    <p style="font-size:0.8rem;margin:0.3rem 0 0;opacity:0.7">
                        Format: JPG, PNG, JPEG, BMP, WEBP
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with col_res:
            st.subheader("Hasil Prediksi")

            if not uploaded:
                st.markdown("""
                **Cara penggunaan:**
                1. Upload foto gigi di sebelah kiri
                2. Tunggu analisis (~1–3 detik)
                3. Lihat prediksi dan confidence
                4. Sesuaikan threshold di sidebar jika perlu

                ---
                **Kelas yang dideteksi:**
                | Kelas | Deskripsi |
                |-------|-----------|
                | 🦷 **Caries** | Kerusakan gigi akibat bakteri asam |
                | 🌿 **Gingivitas** | Peradangan/infeksi pada gusi |
                """)

            else:
                with st.spinner("🔍 Menganalisis gambar..."):
                    try:
                        arr    = preprocess_image(pil_img)
                        result = run_predict(model, arr, idx_to_class, custom_threshold)
                    except Exception as e:
                        st.error(f"❌ Error prediksi: {e}")
                        st.stop()

                label      = result["label"]
                confidence = result["confidence"]
                raw_prob   = result["raw_prob"]
                risk_text  = result["risk_text"]
                risk_color = result["risk_color"]
                risk_icon  = result["risk_icon"]

                # Card hasil
                is_caries  = "aries" in label.lower()
                card_class = "caries" if is_caries else "gingivitas"
                icon       = "🦷" if is_caries else "🌿"

                st.markdown(f"""
                <div class="result-card {card_class}">
                    <p class="result-label">{icon} {label}</p>
                    <p class="result-sub">
                        Confidence: <strong>{confidence:.1%}</strong>
                        &nbsp;·&nbsp;
                        Risk: <strong style="color:{risk_color}">{risk_icon} {risk_text}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence", f"{confidence:.1%}")
                m2.metric("Raw Prob",   f"{raw_prob:.4f}")
                m3.metric("Threshold",  f"{custom_threshold:.2f}")

                # Progress bar per kelas
                st.markdown("**Distribusi Probabilitas**")
                for idx_cls, cls_name in sorted(idx_to_class.items()):
                    p    = raw_prob if idx_cls == 1 else (1.0 - raw_prob)
                    icon = "🦷" if "aries" in cls_name else "🌿"
                    st.progress(float(p), text=f"{icon} {cls_name}: {p:.1%}")

                # Detail teknis
                with st.expander("🔧 Detail Teknis"):
                    st.json({
                        "label"            : label,
                        "confidence"       : round(confidence, 4),
                        "raw_probability"  : round(raw_prob, 4),
                        "threshold_used"   : custom_threshold,
                        "pred_class_idx"   : result["pred_idx"],
                        "preprocessing"    : "mobilenet_v2.preprocess_input [-1, 1]",
                        "model_base"       : meta.get("model_base", "MobileNetV2"),
                        "input_size"       : "224x224",
                    })

                # Disclaimer
                st.markdown("""
                <div class="disclaimer">
                ⚕️ <strong>Disclaimer Medis:</strong>
                Hasil ini bersifat <strong>informatif</strong> dan
                <strong>tidak menggantikan diagnosis dokter gigi profesional</strong>.
                Selalu konsultasikan kondisi gigi Anda dengan tenaga medis yang kompeten.
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — Batch Prediction
    # ══════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Prediksi Banyak Gambar Sekaligus")
        st.caption(
            f"Threshold aktif: `{custom_threshold:.2f}` — "
            "ubah di sidebar jika perlu sebelum memulai batch."
        )

        batch_files = st.file_uploader(
            "Upload gambar (bisa pilih banyak sekaligus — Ctrl+klik)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            key="batch",
        )

        if batch_files:
            st.info(f"**{len(batch_files)} gambar** siap diproses.")

            if st.button("🚀 Jalankan Prediksi Batch", type="primary", use_container_width=True):
                results_list = []
                bar    = st.progress(0, text="Memulai...")
                status = st.empty()

                for i, f in enumerate(batch_files):
                    status.text(f"⏳ Memproses: {f.name} ({i+1}/{len(batch_files)})")
                    try:
                        img = Image.open(f).convert("RGB")
                        arr = preprocess_image(img)
                        res = run_predict(model, arr, idx_to_class, custom_threshold)
                        results_list.append({
                            "No"        : i + 1,
                            "Nama File" : f.name,
                            "Prediksi"  : res["label"],
                            "Confidence": f"{res['confidence']:.2%}",
                            "Raw Prob"  : round(res["raw_prob"], 4),
                            "Risk"      : f"{res['risk_icon']} {res['risk_text']}",
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
                            "Status"    : f"❌ {str(e)[:60]}",
                        })

                    bar.progress((i + 1) / len(batch_files), text=f"{i+1}/{len(batch_files)}")

                status.success("✅ Selesai!")

                df = pd.DataFrame(results_list)

                # Ringkasan
                st.markdown("#### Ringkasan")
                valid = df[df["Prediksi"] != "ERROR"]
                summary_cols = st.columns(2 + len(idx_to_class))
                summary_cols[0].metric("Total", len(df))
                summary_cols[1].metric("Berhasil", len(valid))
                for j, (_, cls_name) in enumerate(sorted(idx_to_class.items())):
                    count = len(valid[valid["Prediksi"] == cls_name])
                    summary_cols[2 + j].metric(cls_name, count)

                # Tabel hasil
                st.markdown("#### Hasil Detail")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Download
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Hasil (.csv)",
                    data=csv,
                    file_name="dental_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        else:
            st.markdown("""
            **Cara pakai:**
            1. Klik tombol upload di atas
            2. Pilih banyak file sekaligus (tahan **Ctrl** lalu klik)
            3. Klik **Jalankan Prediksi Batch**
            4. Download hasilnya sebagai **CSV**
            """)

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — Model Info
    # ══════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Informasi Model")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Arsitektur")
            st.code("""
MobileNetV2  (pretrained ImageNet)
  │  frozen pada Fase 1
  ↓
GlobalAveragePooling2D
Dense(128)          ← tanpa aktivasi dulu
BatchNormalization
Activation('relu')
Dropout(0.4)
Dense(1, sigmoid)   ← binary output
            """, language="text")

            st.markdown("#### Pipeline Training")
            st.markdown("""
| Fase | Keterangan |
|------|-----------|
| **Fase 1** | Feature Extraction — base frozen, LR = 1e-3 |
| **Fase 2** | Fine-tuning — 30 layer unfrozen, LR = 1e-5 |
| **Augmentasi** | flip, brightness, contrast, rot90 |
| **Preprocessing** | `preprocess_input` MobileNetV2 ([-1, 1]) |
| **Class weight** | balanced (atasi imbalance) |
| **Threshold** | dicari dari val set via weighted F1 |
            """)

            st.markdown("#### Dataset")
            caries_count = meta.get("dataset_caries", "~800")
            gingi_count  = meta.get("dataset_gingivitas", "~732")
            st.markdown(f"""
- 🦷 **Caries** : {caries_count} gambar
- 🌿 **Gingivitas** : {gingi_count} gambar
- **Split** : 70% train · 15% val · 15% test
- **Seed** : `{meta.get('seed', 42)}`
            """)

        with col_b:
            st.markdown("#### Metadata Model")
            display_meta = {
                k: v for k, v in meta.items()
                if k not in ("class_map",)  # sembunyikan class_map agar ringkas
            }
            st.json(display_meta)

            st.markdown("#### Performa")
            acc = meta.get("test_accuracy")
            auc = meta.get("test_auc")
            if acc or auc:
                pc1, pc2 = st.columns(2)
                if acc:
                    pc1.metric("Test Accuracy", f"{acc:.4f}", f"{acc*100:.1f}%")
                if auc:
                    pc2.metric("Test AUC", f"{auc:.4f}")
            else:
                st.caption("Metrik belum tersedia di metadata.json")

        st.markdown("---")
        st.warning("""
**⚠️ Catatan penting — val_accuracy = 1.0:**

Model menunjukkan akurasi sempurna sejak epoch awal karena dataset kecil (~1500 gambar)
dan MobileNetV2 sangat powerful untuk tugas ini.

Sebelum digunakan secara klinis, **wajib divalidasi** dengan gambar dari
klinik/sumber berbeda yang belum pernah dilihat model.
        """)

    # ── FOOTER ─────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "🦷 Dental Disease Classifier · "
        "MobileNetV2 Transfer Learning · "
        "Universitas Amikom Yogyakarta · "
        "Dibuat untuk keperluan akademis"
    )


if __name__ == "__main__":
    main()
