import streamlit as st
import cv2
import numpy as np
from skimage import measure
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Smart Cell Analyzer", layout="wide")
st.title("🔬 Smart Cell Analyzer")

def process_image(image, min_size=1, max_size=1000, threshold=30, watershed_dist=0):
    """Processamento inteligente com watershed opcional"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Operações morfológicas
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Watershed opcional (se distância > 0)
    if watershed_dist > 0:
        dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, watershed_dist*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(cleaned, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image, markers)
        markers[markers == -1] = 0
    else:
        _, markers = cv2.connectedComponents(cleaned)
    
    # Analisar marcadores
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

# Interface
with st.sidebar:
    st.header("1. Upload Images")
    nucleus_img = st.file_uploader("Nucleus (Blue)", type=["png", "jpg", "tif"])
    red_img = st.file_uploader("Red Channel", type=["png", "jpg", "tif"])
    green_img = st.file_uploader("Green Channel", type=["png", "jpg", "tif"])

# Parâmetros do núcleo
with st.sidebar:
    st.header("2. Analysis Settings")
    min_size = st.slider("Min cell size (px)", 1, 200, 10)
    max_size = st.slider("Max cell size (px)", 50, 1000, 500)
    threshold = st.slider("Base threshold", 0, 255, 30)
    watershed_dist = st.slider("Watershed separation", 0.0, 1.0, 0.5, 0.1,
                             help="0 = disabled, higher values separate cells more aggressively")

if nucleus_img:
    # Processar núcleo
    nucleus = np.array(Image.open(nucleus_img))
    nucleus_bgr = cv2.cvtColor(nucleus, cv2.COLOR_RGB2BGR)
    
    # Detectar células
    cells = process_image(nucleus_bgr, min_size, max_size, threshold, watershed_dist)
    
    # Visualização
    st.header("Cell Detection")
    display_img = nucleus.copy()
    
    for (cX, cY, cnt) in cells:
        cv2.drawContours(display_img, [cnt], -1, (255, 0, 0), 1)  # Contorno azul
        cv2.circle(display_img, (cX, cY), 2, (0, 255, 255), -1)   # Centro amarelo
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(nucleus, caption="Original", use_column_width=True)
    with col2:
        st.image(display_img, caption=f"Detected: {len(cells)} cells", use_column_width=True)
    
    # Análise dos outros canais
    if red_img or green_img:
        st.header("Channel Analysis")
        results = []
        
        for channel, name, color in zip([red_img, green_img], ["Red", "Green"], [(0,0,255), (0,255,0)]):
            if not channel:
                continue
                
            # Processar canal
            channel_arr = np.array(Image.open(channel))
            channel_bgr = cv2.cvtColor(channel_arr, cv2.COLOR_RGB2BGR)
            
            # Detectar células positivas
            positive_cells = process_image(channel_bgr, min_size, max_size, threshold, watershed_dist)
            
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
                "Total Cells": len(cells),
                "Percentage": (len(positive_cells)/len(cells))*100 if len(cells)>0 else 0
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
