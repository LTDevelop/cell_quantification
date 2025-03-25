import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# Configurações do App
st.set_page_config(page_title="Adaptive Cell Analyzer", layout="wide")
st.title("🔬 Adaptive Cell Detection")

def adaptive_cell_detection(image, min_size=10, max_size=500, block_size=151, C=5):
    """Detecção adaptativa que funciona em regiões densas e esparsas"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold adaptativo (resolve variações de iluminação)
    thresh = cv2.adaptiveThreshold(gray, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 
                                 block_size, C)
    
    # Pós-processamento
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Encontrar componentes conectados
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar por tamanho e forma
    cells = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_size < area < max_size:
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            
            if 0.7 < circularity < 1.3:  # Filtro de forma oval
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cells.append((cX, cY, cnt))
    
    return cells

# Interface
with st.sidebar:
    st.header("1. Upload Images")
    nucleus_img = st.file_uploader("Nucleus (Blue)", type=["png", "jpg", "tif"])
    
    st.header("2. Adaptive Settings")
    min_size = st.slider("Min cell size (px)", 1, 100, 10)
    max_size = st.slider("Max cell size (px)", 50, 1000, 500)
    block_size = st.slider("Adaptive block size", 11, 301, 151, step=10,
                          help="Tamanho da região para cálculo adaptativo (ímpar)")
    C = st.slider("Threshold adjustment", -50, 50, 5,
                 help="Ajuste fino do threshold (valores mais altos = menos sensível)")

if nucleus_img:
    # Processar imagem
    nucleus = np.array(Image.open(nucleus_img))
    nucleus_bgr = cv2.cvtColor(nucleus, cv2.COLOR_RGB2BGR)
    
    # Detecção adaptativa
    cells = adaptive_cell_detection(nucleus_bgr, min_size, max_size, block_size, C)
    
    # Visualização
    st.header("Results")
    display_img = nucleus.copy()
    
    # Desenhar contornos e centros
    for (cX, cY, cnt) in cells:
        cv2.drawContours(display_img, [cnt], -1, (255, 0, 0), 1)  # Contorno azul
        cv2.circle(display_img, (cX, cY), 2, (0, 255, 255), -1)   # Centro amarelo
    
    # Mostrar comparação
    col1, col2 = st.columns(2)
    with col1:
        st.image(nucleus, caption="Original Image", use_column_width=True)
    with col2:
        st.image(display_img, caption=f"Detected Cells: {len(cells)}", use_column_width=True)
    
    # Dicas de ajuste
    with st.expander("💡 Adjustment Tips"):
        st.markdown("""
        - **Regiões densas**: Aumente o 'block size' (151-251) e diminua 'C' (2-10)
        - **Regiões esparsas**: Diminua o 'block size' (51-151) e aumente 'C' (10-20)
        - **Células pequenas**: Ajuste 'Min cell size' conforme necessário
        """)

else:
    st.info("Please upload an image to start analysis")
