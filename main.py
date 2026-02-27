import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import cv2

from sklearn.datasets import load_digits
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from streamlit_drawable_canvas import st_canvas


# -----------------------------
# Helpers
# -----------------------------
def plot_digits_grid(images, labels, n=20):
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 0.7, 3))
    axes = axes.ravel()
    for i in range(n):
        axes[i].imshow(images[i], cmap="gray", vmin=0, vmax=16)
        axes[i].set_title(str(labels[i]))
        axes[i].axis("off")
    plt.tight_layout()
    return fig


def get_cv(strategy_name: str, k: int, repeats: int, seed: int):
    if strategy_name == "StratifiedKFold":
        return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    if strategy_name == "KFold":
        return KFold(n_splits=k, shuffle=True, random_state=seed)
    if strategy_name == "RepeatedStratifiedKFold":
        return RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed)
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


def build_model(model_name: str, seed: int):
    # 5 modelos (punto 3)
    if model_name == "Naive Bayes (GaussianNB)":
        return GaussianNB()
    if model_name == "kNN":
        return KNeighborsClassifier(n_neighbors=5)
    if model_name == "SVM (RBF)":
        return SVC(kernel="rbf", C=5.0, gamma="scale")
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=250, random_state=seed, n_jobs=-1, max_depth=None
        )
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=seed, max_depth=None)

    raise ValueError("Modelo no soportado.")


def make_pipeline(model, use_pca: bool, pca_components: float | int, seed: int):
    steps = [("scaler", StandardScaler())]
    if use_pca:
        # Si pca_components < 1 => varianza explicada; si int => n componentes
        steps.append(("pca", PCA(n_components=pca_components, random_state=seed)))
    steps.append(("model", model))
    return Pipeline(steps)


# -----------------------------
# DIP: Canvas -> 8x8 digits
# -----------------------------
def dip_canvas_to_digits8x8(canvas_rgba: np.ndarray, debug: bool = False):
    """
    Convierte la imagen RGBA del canvas (fondo blanco, trazo negro) a un vector 64
    con escala similar al dataset digits (0..16).
    Incluye DIP: grayscale, inversión, umbral, recorte por bounding box, centrado,
    suavizado, contraste, resize a 8x8 y normalización.
    """
    if canvas_rgba is None:
        return None, None

    # RGBA -> BGR
    img = canvas_rgba.copy()
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # A veces viene RGBA; tomamos RGB
    rgb = img[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Invertir (queremos dígito en blanco sobre negro para procesar fácil)
    inv = 255 - gray

    # Aumentar contraste (opcional / DIP)
    inv = cv2.equalizeHist(inv)

    # Suavizado (DIP)
    inv_blur = cv2.GaussianBlur(inv, (5, 5), 0)

    # Umbral para detectar el trazo
    _, th = cv2.threshold(inv_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Si no hay nada dibujado:
    if th.sum() == 0:
        return None, {"reason": "empty"}

    # Bounding box del contenido
    ys, xs = np.where(th > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # Recorte
    crop = inv_blur[y0 : y1 + 1, x0 : x1 + 1]

    # Padding para cuadrar + margen
    h, w = crop.shape
    side = max(h, w)
    pad = int(0.25 * side)  # margen extra para no cortar bordes
    side2 = side + 2 * pad

    square = np.zeros((side2, side2), dtype=np.uint8)
    yoff = (side2 - h) // 2
    xoff = (side2 - w) // 2
    square[yoff : yoff + h, xoff : xoff + w] = crop

    # Un toque de suavizado final
    square = cv2.GaussianBlur(square, (3, 3), 0)

    # Resize a 8x8
    resized = cv2.resize(square, (8, 8), interpolation=cv2.INTER_AREA)

    # Normalizar a escala digits ~ 0..16
    # Primero re-escalar 0..255 -> 0..16
    digits_8x8 = (resized.astype(np.float32) / 255.0) * 16.0

    # Ajuste de contraste ligero (DIP) para que el trazo sea más claro
    # Evita que quede muy tenue
    mx = digits_8x8.max()
    if mx > 0:
        digits_8x8 = digits_8x8 * (16.0 / mx)
        digits_8x8 = np.clip(digits_8x8, 0, 16)

    vector64 = digits_8x8.reshape(1, -1)

    debug_pack = None
    if debug:
        debug_pack = {
            "gray": gray,
            "inv": inv,
            "inv_blur": inv_blur,
            "th": th,
            "crop": crop,
            "square": square,
            "resized8": resized,
            "digits8": digits_8x8,
        }

    return vector64, debug_pack


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="MNIST Digits (sklearn) - Streamlit ML", layout="wide")
st.title("🧠 MNIST Digits (sklearn) - Clasificación + PCA + CV + Dibujo con DIP")

# Data
digits = load_digits()
X = digits.data  # (n, 64)
y = digits.target
images = digits.images  # (n, 8, 8)

# Sidebar controls
st.sidebar.header("⚙️ Configuración")

seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

train_pct = st.sidebar.slider("Porcentaje para TRAIN", min_value=50, max_value=95, value=80, step=5)
test_size = 1.0 - (train_pct / 100.0)

use_pca = st.sidebar.checkbox("Usar PCA", value=False)
pca_mode = st.sidebar.selectbox("PCA n_components", ["0.95 (varianza)", "30", "40", "50"], index=0)
pca_components = 0.95 if pca_mode.startswith("0.95") else int(pca_mode)

model_name = st.sidebar.selectbox(
    "Modelo (elige 1)",
    [
        "Naive Bayes (GaussianNB)",
        "kNN",
        "SVM (RBF)",
        "Random Forest",
        "Decision Tree",
    ],
)

st.sidebar.subheader("Validación cruzada (4.1)")
cv_strategy = st.sidebar.selectbox(
    "Estrategia CV",
    ["StratifiedKFold", "KFold", "RepeatedStratifiedKFold"],
    index=0,
)
k = st.sidebar.slider("k folds", 3, 10, 5, 1)
repeats = st.sidebar.slider("repeats (si aplica)", 1, 5, 2, 1)
run_cv = st.sidebar.checkbox("Calcular CV score", value=True)

st.sidebar.subheader("Dibujo")
canvas_size = st.sidebar.slider("Tamaño canvas (px)", 200, 420, 280, 20)
stroke_width = st.sidebar.slider("Grosor del trazo", 8, 40, 18, 1)
show_dip_debug = st.sidebar.checkbox("Mostrar debug DIP", value=False)

# Layout
colA, colB = st.columns([1.1, 1.2])

with colA:
    st.subheader("1) Verificar calidad de los datos")
    st.write("Dataset: `sklearn.datasets.load_digits()` (imágenes 8×8, valores ~0..16).")
    st.write(f"- Muestras: **{X.shape[0]}**  | Features: **{X.shape[1]}**  | Clases: **{len(np.unique(y))}**")

    # calidad: nulos, rangos, distribución
    dfq = pd.DataFrame(X)
    st.write(f"- Valores nulos en X: **{int(dfq.isna().sum().sum())}**")
    st.write(f"- Rango aproximado en X: **[{X.min():.1f}, {X.max():.1f}]**")
    counts = pd.Series(y).value_counts().sort_index()
    st.bar_chart(counts)

    st.subheader("Muestras rápidas")
    fig = plot_digits_grid(images, y, n=20)
    st.pyplot(fig, clear_figure=True)

with colB:
    st.subheader("2-4) Entrenamiento / Test + PCA + Modelo")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    st.write(f"Split: Train **{len(X_train)}** ({train_pct}%) | Test **{len(X_test)}** ({int(100-train_pct)}%)")

    model = build_model(model_name, seed=seed)
    pipe = make_pipeline(model, use_pca=use_pca, pca_components=pca_components, seed=seed)

    # CV
    if run_cv:
        cv = get_cv(cv_strategy, k=k, repeats=repeats, seed=seed)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        st.write(
            f"✅ CV ({cv_strategy}) accuracy: **{cv_scores.mean():.4f}** ± **{cv_scores.std():.4f}**"
        )

    # Fit
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    st.metric("Accuracy (Train)", f"{acc_train:.4f}")
    st.metric("Accuracy (Test)", f"{acc_test:.4f}")

    st.subheader("5) Matriz de confusión (Test)")
    cm = confusion_matrix(y_test, y_pred_test, labels=np.arange(10))
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(ax=ax2, cmap="Blues", colorbar=False)
    plt.tight_layout()
    st.pyplot(fig2, clear_figure=True)

    with st.expander("Reporte de clasificación (Test)"):
        st.text(classification_report(y_test, y_pred_test))

# -----------------------------
# Drawing + DIP + Prediction
# -----------------------------
st.divider()
st.subheader("✍️ Dibuja un dígito (0-9) y el modelo lo reconoce (con DIP a 8×8)")

c1, c2 = st.columns([1.0, 1.0])

with c1:
    st.write("Dibuja en negro sobre fondo blanco. Intenta que quede centrado.")
    canvas = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=stroke_width,
        stroke_color="rgba(0,0,0,1)",
        background_color="rgba(255,255,255,1)",
        width=canvas_size,
        height=canvas_size,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("🧹 Limpiar"):
        st.rerun()

with c2:
    st.write("Procesamiento DIP → 8×8 y predicción")

    if canvas.image_data is not None:
        vector64, dbg = dip_canvas_to_digits8x8(canvas.image_data, debug=show_dip_debug)

        if vector64 is None:
            st.warning("No detecté un trazo claro. Dibuja un dígito más grande y centrado 🙂")
        else:
            # Predict
            pred = int(pipe.predict(vector64)[0])
            probs = None
            if hasattr(pipe.named_steps["model"], "predict_proba"):
                probs = pipe.predict_proba(vector64)[0]

            st.success(f"✅ Predicción: **{pred}**")

            # Mostrar 8x8 resultante
            img8 = vector64.reshape(8, 8)
            fig3, ax3 = plt.subplots(figsize=(3.2, 3.2))
            ax3.imshow(img8, cmap="gray", vmin=0, vmax=16)
            ax3.set_title("Imagen 8×8 (post-DIP)")
            ax3.axis("off")
            st.pyplot(fig3, clear_figure=True)

            # Si hay probabilidades, mostrar barras
            if probs is not None:
                st.write("Probabilidades:")
                st.bar_chart(pd.Series(probs, index=list(range(10))))

            # Debug DIP
            if show_dip_debug and isinstance(dbg, dict) and "digits8" in dbg:
                st.write("Debug DIP (algunas etapas):")
                dcol1, dcol2, dcol3 = st.columns(3)

                def show_gray(img, title):
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(img, cmap="gray")
                    ax.set_title(title)
                    ax.axis("off")
                    st.pyplot(fig, clear_figure=True)

                with dcol1:
                    show_gray(dbg["inv_blur"], "Invert + Blur")
                    show_gray(dbg["th"], "Threshold")
                with dcol2:
                    show_gray(dbg["crop"], "Crop")
                    show_gray(dbg["square"], "Square+Pad")
                with dcol3:
                    show_gray(dbg["resized8"], "Resize 8×8 (0..255)")
                    show_gray(dbg["digits8"], "8×8 (0..16)")

st.caption(
    "Nota: `load_digits` no es el MNIST 28×28, sino el dataset clásico de dígitos 8×8 de sklearn."
)
