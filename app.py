import streamlit as st
import cv2
import numpy as np
from skimage import measure
from skimage.segmentation import watershed
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Ultimate Cell Analyzer", layout="wide")
st.title("🔬 Ultimate Cell Analyzer")

def apply_watershed(image, min_distance=10):
    """Aplica watershed para separar células sobrepostas"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remover ruído
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Área de fundo
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Transformada de distância
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Área desconhecida
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marcadores
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    
    # Aplicar watershed
    markers = cv2.watershed(image, markers)
    markers[markers == -1] = 0
    
    return markers

def detect_cells(image, min_size=1, max_size=1000, threshold=30, use_watershed=False):
    """Detecta células com ou sem watershed"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    if use_watershed:
        markers = apply_watershed(image)
        cells = []
        for i in np.unique(markers):
            if i == 0:  # Fundo
                continue
            mask = np.where(markers == i, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = contours[0]
                area = cv2.contourArea(cnt)
                if min_size < area < max_size:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cells.append((cX, cY, cnt))
        return cells
    else:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cells = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_size < area < max_size:
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
    red_img = st.file_uploader("Red Channel", type=["png", "jpg", "tif"])
    green_img = st.file_uploader("Green Channel", type=["png", "jpg", "tif"])

# Parâmetros do núcleo
with st.sidebar:
    st.header("2. Nucleus Settings")
    min_size = st.slider("Min nucleus size (px)", 1, 200, 50)
    max_size = st.slider("Max nucleus size (px)", 100, 1000, 500)
    blue_thresh = st.slider("Blue threshold", 0, 255, 30)
    use_watershed = st.checkbox("Use Watershed", value=False)

if nucleus_img:
    # Processar núcleo
    nucleus = np.array(Image.open(nucleus_img))
    nucleus_bgr = cv2.cvtColor(nucleus, cv2.COLOR_RGB2BGR)
    
    # Detectar núcleos
    nuclei = detect_cells(nucleus_bgr, min_size, max_size, blue_thresh, use_watershed)
    
    # Visualização
    st.header("Nucleus Detection")
    nucleus_vis = nucleus.copy()
    
    for (cX, cY, cnt) in nuclei:
        cv2.drawContours(nucleus_vis, [cnt], -1, (255, 0, 0), 1)  # Contorno azul
        cv2.circle(nucleus_vis, (cX, cY), 2, (0, 255, 255), -1)   # Centro amarelo
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(nucleus, caption="Original", use_column_width=True)
    with col2:
        st.image(nucleus_vis, caption=f"Detected: {len(nuclei)} nuclei", use_column_width=True)
    
    # Análise dos outros canais
    if red_img or green_img:
        st.header("Channel Analysis")
        results = []
        
        for channel, name, color in zip([red_img, green_img], ["Red", "Green"], [(0,0,255), (0,255,0)]):
            if not channel:
                continue
                
            # Configurações do canal
            with st.sidebar:
                st.subheader(f"{name} Settings")
                ch_thresh = st.slider(f"{name} threshold", 0, 255, 30, key=f"{name}_thresh")
                ch_min_size = st.slider(f"Min {name} size", 1, 200, 10, key=f"{name}_min")
                ch_max_size = st.slider(f"Max {name} size", 50, 1000, 500, key=f"{name}_max")
                ch_watershed = st.checkbox(f"Watershed for {name}", value=False, key=f"{name}_ws")
            
            # Processar canal
            channel_arr = np.array(Image.open(channel))
            channel_bgr = cv2.cvtColor(channel_arr, cv2.COLOR_RGB2BGR)
            
            # Detectar células positivas
            positive_cells = detect_cells(channel_bgr, ch_min_size, ch_max_size, ch_thresh, ch_watershed)
            
            # Visualização
            channel_vis = channel_arr.copy()
            for (cX, cY, cnt) in positive_cells:
                cv2.drawContours(channel_vis, [cnt], -1, color, 1)
                cv2.circle(channel_vis, (cX, cY), 2, (255, 255, 0), -1)
            
            # Resultados
            cols = st.columns(2)
            with cols[0]:
                st.image(channel_arr, caption=f"Original {name}", use_column_width=True)
            with cols[1]:
                st.image(channel_vis, caption=f"{name} Positives: {len(positive_cells)}", use_column_width=True)
            
            results.append({
                "Channel": name,
                "Positive Cells": len(positive_cells),
                "Total Cells": len(nuclei),
                "Percentage": (len(positive_cells)/len(nuclei))*100 if len(nuclei)>0 else 0
            })
        
        # Tabela de resultados
        st.header("Quantitative Results")
        df = pd.DataFrame(results)
        st.dataframe(df, hide_index=True)
        
        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "cell_counts.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload at least the nucleus image")
