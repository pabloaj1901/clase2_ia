import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Modelos
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")
st.title("🔢 Clasificador de Dígitos MNIST - SKLearn")
st.markdown("Clasificador multiclase de 10 salidas (dígitos 0-9)")

# ============================================================
# 1. CARGAR Y VERIFICAR CALIDAD DE DATOS
# ============================================================
st.header("1. Verificación de Calidad de Datos")

digits = load_digits()
X, y = digits.data, digits.target

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de muestras", X.shape[0])
    st.metric("Características (píxeles)", X.shape[1])
with col2:
    st.metric("Clases únicas", len(np.unique(y)))
    st.metric("Valores nulos", int(np.isnan(X).sum()))
with col3:
    st.metric("Valores infinitos", int(np.isinf(X).sum()))
    st.metric("Rango de píxeles", f"{X.min():.0f} - {X.max():.0f}")

# Distribución de clases
st.subheader("Distribución de clases")
fig_dist, ax_dist = plt.subplots(figsize=(10, 3))
unique, counts = np.unique(y, return_counts=True)
ax_dist.bar(unique, counts, color='steelblue')
ax_dist.set_xlabel("Dígito")
ax_dist.set_ylabel("Cantidad")
ax_dist.set_xticks(range(10))
st.pyplot(fig_dist)

# Muestra de imágenes
st.subheader("Muestra de imágenes por clase")
fig_samples, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = np.where(y == i)[0][0]
    ax.imshow(digits.images[idx], cmap='gray_r')
    ax.set_title(f"Dígito: {i}")
    ax.axis('off')
plt.tight_layout()
st.pyplot(fig_samples)

# ============================================================
# 2. CONFIGURACIÓN: PCA, SPLIT, MODELO
# ============================================================
st.header("2. Configuración del Experimento")

col_cfg1, col_cfg2 = st.columns(2)

with col_cfg1:
    usar_pca = st.checkbox("Aplicar PCA (reducción de dimensionalidad)", value=False)
    n_components = None
    if usar_pca:
        n_components = st.slider("Número de componentes PCA", 2, 60, 30)
    
    test_size = st.slider("Porcentaje de datos para TEST (%)", 10, 50, 20, step=5)

with col_cfg2:
    modelo_nombre = st.selectbox("Selecciona el modelo de clasificación", [
        "Naive Bayes (GaussianNB)",
        "K-Nearest Neighbors (KNN)",
        "Support Vector Machine (SVM)",
        "Random Forest (RF)",
        "Decision Tree (DT)",
        "Red Neuronal / MLP (ANN)"
    ])

    estrategia_cv = st.selectbox("Estrategia de Validación Cruzada", [
        "K-Fold (5)",
        "K-Fold (10)",
        "Stratified K-Fold (5)",
        "Stratified K-Fold (10)",
        "Shuffle Split (5 iteraciones)"
    ])

# ============================================================
# 3. ENTRENAR MODELO
# ============================================================
if st.button("🚀 Entrenar Modelo", type="primary"):
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA opcional
    pca = None
    if usar_pca and n_components:
        pca = PCA(n_components=n_components)
        X_processed = pca.fit_transform(X_scaled)
        st.info(f"PCA aplicado: {X.shape[1]} → {n_components} componentes. "
                f"Varianza explicada: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    else:
        X_processed = X_scaled
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size/100, random_state=42, stratify=y
    )
    
    st.write(f"**Train:** {X_train.shape[0]} muestras | **Test:** {X_test.shape[0]} muestras")
    
    # Seleccionar modelo
    modelos = {
        "Naive Bayes (GaussianNB)": GaussianNB(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine (SVM)": SVC(kernel='rbf', gamma='scale', random_state=42),
        "Random Forest (RF)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree (DT)": DecisionTreeClassifier(random_state=42),
        "Red Neuronal / MLP (ANN)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    }
    
    modelo = modelos[modelo_nombre]
    
    # Entrenar
    with st.spinner(f"Entrenando {modelo_nombre}..."):
        modelo.fit(X_train, y_train)
    
    # Predicciones
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # ============================================================
    # MÉTRICAS
    # ============================================================
    st.header("3. Resultados de Desempeño")
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test, average='weighted')
    rec_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Accuracy Train", f"{acc_train:.4f}")
    col_m2.metric("Accuracy Test", f"{acc_test:.4f}")
    col_m3.metric("Precision Test", f"{prec_test:.4f}")
    col_m4.metric("Recall Test", f"{rec_test:.4f}")
    col_m5.metric("F1-Score Test", f"{f1_test:.4f}")
    
    # Gráfica comparativa Train vs Test
    st.subheader("Comparación Train vs Test")
    fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
    metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    prec_train = precision_score(y_train, y_pred_train, average='weighted')
    rec_train = recall_score(y_train, y_pred_train, average='weighted')
    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    
    vals_train = [acc_train, prec_train, rec_train, f1_train]
    vals_test = [acc_test, prec_test, rec_test, f1_test]
    
    x_pos = np.arange(len(metricas_nombres))
    width = 0.35
    bars1 = ax_comp.bar(x_pos - width/2, vals_train, width, label='Train', color='steelblue')
    bars2 = ax_comp.bar(x_pos + width/2, vals_test, width, label='Test', color='coral')
    ax_comp.set_ylim(0, 1.1)
    ax_comp.set_xticks(x_pos)
    ax_comp.set_xticklabels(metricas_nombres)
    ax_comp.legend()
    ax_comp.set_title(f"Desempeño: {modelo_nombre}")
    
    for bar in bars1:
        ax_comp.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax_comp.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    
    st.pyplot(fig_comp)
    
    # Matriz de confusión
    st.subheader("Matriz de Confusión (Test)")
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10),
                yticklabels=range(10), ax=ax_cm)
    ax_cm.set_xlabel("Predicción")
    ax_cm.set_ylabel("Real")
    ax_cm.set_title("Matriz de Confusión")
    st.pyplot(fig_cm)
    
    # Reporte de clasificación
    st.subheader("Reporte de Clasificación")
    report = classification_report(y_test, y_pred_test, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4))
    
    # ============================================================
    # 4.1 VALIDACIÓN CRUZADA
    # ============================================================
    st.header("4. Validación Cruzada")
    
    cv_strategies = {
        "K-Fold (5)": KFold(n_splits=5, shuffle=True, random_state=42),
        "K-Fold (10)": KFold(n_splits=10, shuffle=True, random_state=42),
        "Stratified K-Fold (5)": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "Stratified K-Fold (10)": StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        "Shuffle Split (5 iteraciones)": ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    }
    
    cv = cv_strategies[estrategia_cv]
    
    # Recrear modelo limpio para CV
    modelo_cv = modelos[modelo_nombre].__class__(**modelos[modelo_nombre].get_params())
    
    with st.spinner("Ejecutando validación cruzada..."):
        cv_scores = cross_val_score(modelo_cv, X_processed, y, cv=cv, scoring='accuracy')
    
    col_cv1, col_cv2, col_cv3 = st.columns(3)
    col_cv1.metric("CV Media", f"{cv_scores.mean():.4f}")
    col_cv2.metric("CV Desv. Estándar", f"{cv_scores.std():.4f}")
    col_cv3.metric("CV Folds", len(cv_scores))
    
    fig_cv, ax_cv = plt.subplots(figsize=(8, 3))
    ax_cv.bar(range(1, len(cv_scores)+1), cv_scores, color='teal')
    ax_cv.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Media: {cv_scores.mean():.4f}')
    ax_cv.set_xlabel("Fold")
    ax_cv.set_ylabel("Accuracy")
    ax_cv.set_title(f"Validación Cruzada - {estrategia_cv}")
    ax_cv.legend()
    ax_cv.set_ylim(0, 1.1)
    st.pyplot(fig_cv)
    
    # Guardar modelo y procesadores en session_state
    st.session_state['modelo'] = modelo
    st.session_state['scaler'] = scaler
    st.session_state['pca'] = pca
    st.session_state['usar_pca'] = usar_pca
    st.session_state['modelo_entrenado'] = True
    
    st.success("✅ Modelo entrenado y listo para predicción.")

# ============================================================
# 5. DIBUJAR DÍGITO PARA RECONOCERLO
# ============================================================
st.header("5. Dibuja un Dígito para Reconocerlo")

if not st.session_state.get('modelo_entrenado', False):
    st.warning("⚠️ Primero entrena un modelo arriba para poder hacer predicciones.")
else:
    st.markdown("Dibuja un dígito (0-9) en el canvas de abajo y presiona **Predecir**.")
    
    def preprocess_canvas_image(image_data):
        """
        Pipeline de Digital Image Processing (DIP) para convertir
        la imagen del canvas en una imagen compatible con sklearn digits (8x8, 0-16).
        
        Etapas:
        1. Conversión a escala de grises
        2. Binarización con umbral adaptativo (Otsu)
        3. Operaciones morfológicas (cierre para rellenar huecos)
        4. Detección de bounding box y recorte
        5. Centrado por centro de masa
        6. Resize a 8x8 con antialiasing
        7. Suavizado gaussiano (simular estilo sklearn digits)
        8. Normalización de contraste al rango 0-16
        9. Ajuste de distribución de intensidad
        """
        from scipy.ndimage import gaussian_filter, center_of_mass, zoom
        from scipy.ndimage import binary_dilation, binary_closing
        
        img = image_data.astype(np.uint8)
        
        # ---- ETAPA 1: Escala de grises ----
        img_gray = np.mean(img[:, :, :3], axis=2).astype(np.float64)
        
        # ---- ETAPA 2: Binarización (Otsu simplificado) ----
        threshold = 30
        if img_gray.max() > 0:
            # Umbral adaptativo: usar percentil de los píxeles no-cero
            nonzero = img_gray[img_gray > 10]
            if len(nonzero) > 0:
                threshold = max(30, np.percentile(nonzero, 15))
        
        binary_mask = img_gray > threshold
        
        if binary_mask.sum() == 0:
            return None  # No se dibujó nada
        
        # ---- ETAPA 3: Operaciones morfológicas ----
        # Cierre morfológico para rellenar huecos en los trazos
        struct = np.ones((5, 5), dtype=bool)
        binary_closed = binary_closing(binary_mask, structure=struct, iterations=2)
        # Dilatación leve para engrosar trazos finos
        struct_dilate = np.ones((3, 3), dtype=bool)
        binary_dilated = binary_dilation(binary_closed, structure=struct_dilate, iterations=1)
        
        # Aplicar máscara morfológica a la imagen original
        img_processed = img_gray * binary_dilated.astype(np.float64)
        
        # ---- ETAPA 4: Bounding box y recorte ----
        coords = np.argwhere(binary_dilated)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        digit_crop = img_processed[y_min:y_max+1, x_min:x_max+1]
        
        # ---- ETAPA 5: Centrado por centro de masa en cuadrado ----
        h, w = digit_crop.shape
        max_dim = max(h, w)
        
        # Padding del 30% (similar a sklearn digits donde los bordes tienen valores bajos)
        padding = int(max_dim * 0.35)
        square_size = max_dim + 2 * padding
        
        square_img = np.zeros((square_size, square_size), dtype=np.float64)
        
        # Calcular centro de masa del dígito recortado
        cy, cx = center_of_mass(digit_crop)
        
        # Colocar el dígito de forma que su centro de masa quede en el centro del cuadrado
        target_cy = square_size // 2
        target_cx = square_size // 2
        
        y_offset = int(target_cy - cy)
        x_offset = int(target_cx - cx)
        
        # Clamp offsets para no salirse del cuadrado
        y_offset = max(0, min(y_offset, square_size - h))
        x_offset = max(0, min(x_offset, square_size - w))
        
        square_img[y_offset:y_offset+h, x_offset:x_offset+w] = digit_crop
        
        # ---- ETAPA 6: Resize a 8x8 con antialiasing ----
        img_pil = Image.fromarray(square_img.astype(np.uint8))
        img_8x8 = img_pil.resize((8, 8), Image.LANCZOS)
        img_array = np.array(img_8x8, dtype=np.float64)
        
        # ---- ETAPA 7: Suavizado gaussiano ----
        # Las imágenes de sklearn digits tienen un suavizado natural
        # por cómo fueron originalmente creadas (escritura suave a baja resolución)
        img_array = gaussian_filter(img_array, sigma=0.6)
        
        # ---- ETAPA 8: Normalización de contraste al rango 0-16 ----
        if img_array.max() > 0:
            img_array = (img_array / img_array.max()) * 16.0
        
        # ---- ETAPA 9: Ajuste de distribución de intensidad ----
        # Las imágenes del dataset tienen:
        # - Media global ~4.88, Std ~6.0
        # - Bordes (fila/col 0 y 7) con valores bajos (~2.3 promedio)
        # - Centro de masa cerca de (3.5, 3.5) en la grid 8x8
        # Atenuar bordes para simular el estilo del dataset
        border_attenuation = np.ones((8, 8), dtype=np.float64)
        border_attenuation[0, :] *= 0.4
        border_attenuation[7, :] *= 0.4
        border_attenuation[:, 0] *= 0.4
        border_attenuation[:, 7] *= 0.4
        # Esquinas aún más atenuadas
        for r, c in [(0,0),(0,7),(7,0),(7,7)]:
            border_attenuation[r, c] *= 0.3
        
        img_array *= border_attenuation
        
        # Re-normalizar al rango 0-16 después de atenuación
        if img_array.max() > 0:
            img_array = (img_array / img_array.max()) * 16.0
        
        # Redondear a enteros como en el dataset original
        img_array = np.round(img_array)
        
        return img_array
    
    def get_dip_stages(image_data):
        """
        Retorna las etapas intermedias del DIP para visualización.
        """
        from scipy.ndimage import gaussian_filter, center_of_mass
        from scipy.ndimage import binary_dilation, binary_closing
        
        stages = {}
        img = image_data.astype(np.uint8)
        
        # Etapa 1: Escala de grises
        img_gray = np.mean(img[:, :, :3], axis=2).astype(np.float64)
        stages['1. Escala de grises'] = img_gray.copy()
        
        # Etapa 2: Binarización
        threshold = 30
        nonzero = img_gray[img_gray > 10]
        if len(nonzero) > 0:
            threshold = max(30, np.percentile(nonzero, 15))
        binary_mask = img_gray > threshold
        stages['2. Binarización'] = binary_mask.astype(np.float64) * 255
        
        # Etapa 3: Morfología
        struct = np.ones((5, 5), dtype=bool)
        binary_closed = binary_closing(binary_mask, structure=struct, iterations=2)
        struct_dilate = np.ones((3, 3), dtype=bool)
        binary_dilated = binary_dilation(binary_closed, structure=struct_dilate, iterations=1)
        stages['3. Morfología'] = binary_dilated.astype(np.float64) * 255
        
        # Etapa 4: Imagen filtrada
        img_processed = img_gray * binary_dilated.astype(np.float64)
        stages['4. Filtrada'] = img_processed.copy()
        
        return stages
    
    def predict_digit(img_array):
        """Realiza la predicción con el modelo entrenado."""
        img_flat = img_array.flatten().reshape(1, -1)
        
        modelo = st.session_state['modelo']
        scaler = st.session_state['scaler']
        pca = st.session_state['pca']
        usar_pca = st.session_state['usar_pca']
        
        img_scaled = scaler.transform(img_flat)
        if usar_pca and pca is not None:
            img_scaled = pca.transform(img_scaled)
        
        prediccion = modelo.predict(img_scaled)
        
        # Probabilidades si están disponibles
        probs = None
        if hasattr(modelo, 'predict_proba'):
            probs = modelo.predict_proba(img_scaled)[0]
        
        return prediccion[0], probs
    
    try:
        from streamlit_drawable_canvas import st_canvas
        
        st.markdown("""
        **Tips para mejor precisión:**
        - Dibuja el dígito **grande y centrado** en el canvas
        - Usa trazos **gruesos**
        - Dibuja de forma **simple**, similar a dígitos impresos
        """)
        
        col_draw, col_result = st.columns([1, 1])
        
        with col_draw:
            stroke = st.slider("Grosor del trazo", 15, 40, 25)
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.3)",
                stroke_width=stroke,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
        
        with col_result:
            if st.button("🔍 Predecir Dígito", type="primary"):
                if canvas_result.image_data is not None:
                    img_array = preprocess_canvas_image(canvas_result.image_data)
                    
                    if img_array is None:
                        st.warning("No se detectó ningún dibujo. Intenta de nuevo.")
                    else:
                        # Mostrar etapas del pipeline DIP
                        stages = get_dip_stages(canvas_result.image_data)
                        
                        st.write("**Pipeline DIP (Digital Image Processing):**")
                        fig_dip, axes_dip = plt.subplots(1, len(stages) + 1, figsize=(14, 3))
                        
                        for idx, (name, stage_img) in enumerate(stages.items()):
                            axes_dip[idx].imshow(stage_img, cmap='gray_r')
                            axes_dip[idx].set_title(name, fontsize=8)
                            axes_dip[idx].axis('off')
                        
                        # Resultado final 8x8
                        axes_dip[-1].imshow(img_array, cmap='gray_r', interpolation='nearest')
                        axes_dip[-1].set_title('5. Final (8x8)', fontsize=8)
                        axes_dip[-1].axis('off')
                        plt.tight_layout()
                        st.pyplot(fig_dip)
                        
                        # Comparar con una muestra real del dataset
                        st.write("**Comparación con muestra real del dataset:**")
                        fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(5, 2.5))
                        axes_cmp[0].imshow(img_array, cmap='gray_r', interpolation='nearest')
                        axes_cmp[0].set_title("Tu dibujo (procesado)", fontsize=9)
                        axes_cmp[0].axis('off')
                        
                        # Predecir primero para mostrar ejemplo del dígito predicho
                        pred, probs = predict_digit(img_array)
                        
                        # Mostrar muestra real del dataset al lado
                        digits_data = load_digits()
                        mask = digits_data.target == pred
                        sample_real = digits_data.images[mask][0]
                        axes_cmp[1].imshow(sample_real, cmap='gray_r', interpolation='nearest')
                        axes_cmp[1].set_title(f"Muestra real (dígito {pred})", fontsize=9)
                        axes_cmp[1].axis('off')
                        plt.tight_layout()
                        st.pyplot(fig_cmp)
                        
                        st.success(f"## 🎯 Dígito predicho: **{pred}**")
                        
                        # Mostrar valores numéricos de la imagen procesada
                        with st.expander("Ver matriz de valores (8x8)"):
                            st.dataframe(pd.DataFrame(img_array).round(1))
                        
                        if probs is not None:
                            fig_prob, ax_prob = plt.subplots(figsize=(6, 3))
                            colors = ['coral' if i != pred else 'steelblue' for i in range(10)]
                            ax_prob.bar(range(10), probs, color=colors)
                            ax_prob.set_xticks(range(10))
                            ax_prob.set_xlabel("Dígito")
                            ax_prob.set_ylabel("Probabilidad")
                            ax_prob.set_title("Probabilidades por clase")
                            for i, p in enumerate(probs):
                                if p > 0.05:
                                    ax_prob.text(i, p + 0.01, f'{p:.2f}', ha='center', fontsize=8)
                            st.pyplot(fig_prob)
                        
                        # Mostrar los 5 dígitos más similares del dataset
                        st.write("**Ejemplos del dataset para el dígito predicho:**")
                        similar_images = digits_data.images[mask][:5]
                        fig_sim, axes_sim = plt.subplots(1, 5, figsize=(10, 2))
                        for i, ax in enumerate(axes_sim):
                            if i < len(similar_images):
                                ax.imshow(similar_images[i], cmap='gray_r', interpolation='nearest')
                            ax.axis('off')
                        plt.suptitle(f"Ejemplos del dígito {pred} en el dataset", fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig_sim)
                else:
                    st.warning("Dibuja algo en el canvas primero.")
    
    except ImportError:
        st.error("No se pudo cargar `streamlit-drawable-canvas`. Instálalo con: `pip install streamlit-drawable-canvas`")
        
        st.markdown("### Alternativa: Sube una imagen de un dígito")
        uploaded_file = st.file_uploader("Sube una imagen de un dígito (PNG/JPG)", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None and st.button("🔍 Predecir Dígito (imagen subida)"):
            img_pil = Image.open(uploaded_file).convert('L')
            img_array_up = np.array(img_pil, dtype=np.float64)
            
            # Aplicar el mismo preprocesamiento de bounding box y centrado
            from scipy.ndimage import gaussian_filter
            threshold = 30
            coords = np.argwhere(img_array_up > threshold)
            if coords.size > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                digit_crop = img_array_up[y_min:y_max+1, x_min:x_max+1]
                h, w = digit_crop.shape
                max_dim = max(h, w)
                padding = int(max_dim * 0.3)
                square_size = max_dim + 2 * padding
                square_img = np.zeros((square_size, square_size), dtype=np.float64)
                y_offset = (square_size - h) // 2
                x_offset = (square_size - w) // 2
                square_img[y_offset:y_offset+h, x_offset:x_offset+w] = digit_crop
                img_pil2 = Image.fromarray(square_img.astype(np.uint8))
                img_resized = img_pil2.resize((8, 8), Image.LANCZOS)
                img_array = np.array(img_resized, dtype=np.float64)
                img_array = gaussian_filter(img_array, sigma=0.5)
                if img_array.max() > 0:
                    img_array = (img_array / img_array.max()) * 16.0
            else:
                img_resized = img_pil.resize((8, 8), Image.LANCZOS)
                img_array = np.array(img_resized, dtype=np.float64)
                if img_array.max() > 0:
                    img_array = (img_array / img_array.max()) * 16.0
            
            fig_up, ax_up = plt.subplots(figsize=(3, 3))
            ax_up.imshow(img_array, cmap='gray_r', interpolation='nearest')
            ax_up.axis('off')
            st.pyplot(fig_up)
            
            pred, probs = predict_digit(img_array)
            st.success(f"## 🎯 Dígito predicho: **{pred}**")
