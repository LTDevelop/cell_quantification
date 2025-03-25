import streamlit as st
import cv2
import numpy as np
from skimage import measure
from skimage.segmentation import watershed
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Smart Cell Analyzer", layout="wide")
st.title("🔬 Smart Cell Detection with Shape Analysis")

def smart_cell_detection(image, min_size=50, intensity_thresh=30, circularity_thresh=0.7):
    """Detecção inteligente considerando forma e intensidade"""
    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Pré-processamento avançado
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    
    # Threshold adaptativo
    thresh = cv2.adaptiveThreshold(equalized, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Operações morfológicas
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar por tamanho e circularidade
    valid_cells = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_size:
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < circularity_thresh:
            continue
            
        # Verificar intensidade média
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        
        if mean_intensity > intensity_thresh:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                valid_cells.append((cX, cY, cnt))
    
    return valid_cells

# Sidebar para uploads
with st.sidebar:
    st.header("1. Upload Images")
    nucleus_img = st.file_uploader("Upload Nucleus (Blue) image", type=["png", "jpg", "tif"])
    channel1_img = st.file_uploader("Upload Channel 1 (Red) image", type=["png", "jpg", "tif"])
    channel2_img = st.file_uploader("Upload Channel 2 (Green) image", type=["png", "jpg", "tif"])

# Processamento principal
if nucleus_img is not None:
    # Carregar imagens
    nucleus = np.array(Image.open(nucleus_img))
    nucleus_bgr = cv2.cvtColor(nucleus, cv2.COLOR_RGB2BGR)
    
    # Configurações na sidebar
    with st.sidebar:
        st.header("2. Detection Settings")
        min_size = st.slider("Min cell size (px)", 20, 200, 50)
        intensity_thresh = st.slider("Intensity threshold", 0, 255, 30)
        circularity = st.slider("Circularity threshold", 0.1, 1.0, 0.7, 0.05)
    
    # Detectar núcleos
    nuclei = smart_cell_detection(nucleus_bgr, min_size, intensity_thresh, circularity)
    
    # Visualização dos núcleos
    st.header("Nucleus Detection")
    nucleus_vis = nucleus.copy()
    
    for (cX, cY, cnt) in nuclei:
        cv2.drawContours(nucleus_vis, [cnt], -1, (255, 0, 0), 1)  # Contorno azul
        cv2.circle(nucleus_vis, (cX, cY), 2, (0, 255, 255), -1)   # Centro amarelo
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(nucleus, caption="Original Nucleus", use_column_width=True)
    with col2:
        st.image(nucleus_vis, caption="Detected Nuclei", use_column_width=True)
    
    # Análise dos canais adicionais
    if channel1_img is not None or channel2_img is not None:
        st.header("Channel Analysis")
        results = []
        
        # Processar cada canal adicional
        for idx, (name, img) in enumerate(zip(["Red", "Green"], [channel1_img, channel2_img])):
            if img is None:
                continue
                
            # Configurações do canal na sidebar
            with st.sidebar:
                st.header(f"{name} Settings")
                ch_intensity = st.slider(f"{name} intensity", 0, 255, 50, key=f"{name}_int")
                ch_sensitivity = st.slider(f"{name} sensitivity", 1, 100, 30, key=f"{name}_sens")
            
            # Processar canal
            channel = np.array(Image.open(img))
            channel_bgr = cv2.cvtColor(channel, cv2.COLOR_RGB2BGR)
            
            # Detectar células positivas
            positive_cells = 0
            channel_vis = channel.copy()
            
            for (cX, cY, cnt) in nuclei:
                # Criar máscara para a célula atual
                mask = np.zeros_like(channel_bgr[:,:,0])
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                # Calcular intensidade média no canal
                mean_intensity = cv2.mean(channel_bgr[:,:,2 if name=="Red" else 1], mask=mask)[0]
                
                if mean_intensity > ch_intensity:
                    positive_cells += 1
                    cv2.drawContours(channel_vis, [cnt], -1, (255, 0, 0), 1)  # Contorno
                    cv2.circle(channel_vis, (cX, cY), 2, (255, 255, 0), -1)    # Centro ciano
            
            # Mostrar resultados
            cols = st.columns(2)
            with cols[0]:
                st.image(channel, caption=f"Original {name}", use_column_width=True)
            with cols[1]:
                st.image(channel_vis, caption=f"{name} Positive Cells", use_column_width=True)
            
            results.append({
                "Channel": name,
                "Positive Cells": positive_cells,
                "Total Cells": len(nuclei),
                "Percentage": (positive_cells / len(nuclei)) * 100 if len(nuclei) > 0 else 0
            })
        
        # Resultados quantitativos
        st.header("Quantitative Results")
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, hide_index=True)
            
            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "cell_analysis.csv",
                mime="text/csv"
            )

else:
    st.info("Please upload at least the nucleus image to start analysis")
