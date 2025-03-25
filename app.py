import streamlit as st
import cv2
import numpy as np
from skimage import measure
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Precision Cell Analyzer", layout="wide")
st.title("🔬 Precision Cell Analysis (Micron-scale)")

# Constantes baseadas em sua informação (50μm = 100px)
MICRONS_PER_PIXEL = 0.5  # 50μm/100px
MIN_CELL_DIAMETER = 4    # μm
MAX_CELL_DIAMETER = 14   # μm

def micron_to_pixels(microns):
    return int(microns / MICRONS_PER_PIXEL)

def detect_cells(image, channel_name):
    """Detecção precisa baseada em tamanho real e forma"""
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
    
    # Filtrar por tamanho real e forma
    valid_cells = []
    min_area = np.pi * (micron_to_pixels(MIN_CELL_DIAMETER/2))**2
    max_area = np.pi * (micron_to_pixels(MAX_CELL_DIAMETER/2))**2
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
            
        # Verificar circularidade (0.7-1.3 para aceitar ovais)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.7 or circularity > 1.3:
            continue
            
        # Verificar intensidade (mínimo 30% do máximo)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        max_intensity = np.max(gray)
        
        if mean_intensity < 0.3 * max_intensity:
            continue
            
        # Se passou em todos os filtros
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            valid_cells.append((cX, cY, cnt))
    
    return valid_cells

# Interface
with st.sidebar:
    st.header("1. Upload Images")
    nucleus_img = st.file_uploader("Nucleus (Blue)", type=["png", "jpg", "tif"])
    red_img = st.file_uploader("Red Channel", type=["png", "jpg", "tif"])
    green_img = st.file_uploader("Green Channel", type=["png", "jpg", "tif"])

if nucleus_img:
    # Processar núcleo
    nucleus = np.array(Image.open(nucleus_img))
    nucleus_bgr = cv2.cvtColor(nucleus, cv2.COLOR_RGB2BGR)
    
    # Detectar células no núcleo
    nuclei = detect_cells(nucleus_bgr, "blue")
    
    # Visualização
    st.header("Nucleus Detection")
    nucleus_vis = nucleus.copy()
    
    for (cX, cY, cnt) in nuclei:
        cv2.drawContours(nucleus_vis, [cnt], -1, (255, 0, 0), 1)
        cv2.circle(nucleus_vis, (cX, cY), 2, (0, 255, 255), -1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(nucleus, caption="Original", use_column_width=True)
    with col2:
        st.image(nucleus_vis, caption=f"Detected: {len(nuclei)} cells", use_column_width=True)
    
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
            
            # Configurações
            with st.sidebar:
                st.subheader(f"{name} Settings")
                intensity_thresh = st.slider(f"Intensity threshold", 0, 255, 30, key=f"{name}_thresh")
                min_overlap = st.slider(f"Min overlap (%)", 0, 100, 30, key=f"{name}_overlap")
            
            # Detecção de células positivas
            positive_cells = 0
            channel_vis = channel_arr.copy()
            
            for (cX, cY, cnt) in nuclei:
                # Máscara para a célula atual
                mask = np.zeros_like(channel_bgr[:,:,0])
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                # Intensidade média no canal específico
                if name == "Red":
                    channel_values = channel_bgr[:,:,2]  # Canal vermelho
                else:
                    channel_values = channel_bgr[:,:,1]  # Canal verde
                
                mean_intensity = cv2.mean(channel_values, mask=mask)[0]
                
                # Se intensidade suficiente e dentro da área celular
                if mean_intensity > intensity_thresh:
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
