import streamlit as st
import cv2
import numpy as np
from skimage import measure
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Precision Cell Counter", layout="wide")
st.title("🔬 Precision Cell Counter")

def detect_nuclei(image, min_size=50, max_size=500, threshold=30):
    """Detecta núcleos com base em tamanho e intensidade"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Operações morfológicas
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar por tamanho
    nuclei = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_size < area < max_size:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                nuclei.append((cX, cY, cnt))
    
    return nuclei

# Interface
with st.sidebar:
    st.header("1. Upload Images")
    nucleus_img = st.file_uploader("Nucleus (Blue)", type=["png", "jpg", "tif"])
    red_img = st.file_uploader("Red Channel", type=["png", "jpg", "tif"])
    green_img = st.file_uploader("Green Channel", type=["png", "jpg", "tif"])

# Parâmetros do núcleo
with st.sidebar:
    st.header("2. Nucleus Settings")
    min_size = st.slider("Min nucleus size (px)", 10, 200, 50)
    max_size = st.slider("Max nucleus size (px)", 100, 1000, 500)
    blue_thresh = st.slider("Blue threshold", 0, 255, 30)

if nucleus_img:
    # Processar núcleo
    nucleus = np.array(Image.open(nucleus_img))
    nucleus_bgr = cv2.cvtColor(nucleus, cv2.COLOR_RGB2BGR)
    
    # Detectar núcleos
    nuclei = detect_nuclei(nucleus_bgr, min_size, max_size, blue_thresh)
    
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
            
            # Processar canal
            channel_arr = np.array(Image.open(channel))
            channel_bgr = cv2.cvtColor(channel_arr, cv2.COLOR_RGB2BGR)
            
            # Detectar células positivas
            positive_cells = 0
            channel_vis = channel_arr.copy()
            
            for (cX, cY, cnt) in nuclei:
                # Máscara para a célula atual
                mask = np.zeros_like(channel_bgr[:,:,0])
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                # Intensidade média no canal
                if name == "Red":
                    ch_values = channel_bgr[:,:,2]  # Canal vermelho
                else:
                    ch_values = channel_bgr[:,:,1]  # Canal verde
                
                mean_intensity = cv2.mean(ch_values, mask=mask)[0]
                
                if mean_intensity > ch_thresh:
                    positive_cells += 1
                    cv2.drawContours(channel_vis, [cnt], -1, color, 1)
                    cv2.circle(channel_vis, (cX, cY), 2, (255, 255, 0), -1)
            
            # Resultados
            cols = st.columns(2)
            with cols[0]:
                st.image(channel_arr, caption=f"Original {name}", use_column_width=True)
            with cols[1]:
                st.image(channel_vis, caption=f"{name} Positives: {positive_cells}", use_column_width=True)
            
            results.append({
                "Channel": name,
                "Positive Cells": positive_cells,
                "Total Cells": len(nuclei),
                "Percentage": (positive_cells/len(nuclei))*100 if len(nuclei)>0 else 0
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
