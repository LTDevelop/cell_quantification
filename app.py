import streamlit as st
import cv2
import numpy as np
from skimage import measure
from skimage.segmentation import watershed
from scipy import ndimage
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Precision Cell Analyzer", layout="wide")
st.title("🔬 Precision Cell Analysis")

def enhanced_segment_cells(image, min_distance=10, min_size=50, threshold=0.5):
    """Segmentação melhorada com pré-processamento"""
    # Converter para escala de cinza e equalizar histograma
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Suavização adaptativa
    blurred = cv2.medianBlur(gray, 5)
    
    # Threshold adaptativo com correção de iluminação
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Operações morfológicas
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Remover pequenos objetos
    cleaned = ndimage.binary_fill_holes(opening)
    cleaned = cleaned.astype(np.uint8) * 255
    
    # Watershed com marcadores
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, threshold*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Encontrar marcadores
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[cleaned == 0] = 0
    
    # Aplicar watershed
    markers = watershed(-dist_transform, markers, mask=cleaned)
    
    # Filtrar por tamanho
    unique, counts = np.unique(markers, return_counts=True)
    for (i, count) in zip(unique, counts):
        if count < min_size and i != 0:
            markers[markers == i] = 0
    
    return markers

def load_channel(name, key_suffix):
    """Interface para upload de cada canal"""
    uploaded = st.file_uploader(f"Upload {name} image", 
                              type=["png", "jpg", "tif"], 
                              key=f"{name}_{key_suffix}")
    if uploaded:
        img = Image.open(uploaded)
        return np.array(img)
    return None

# Sidebar para uploads e configurações
with st.sidebar:
    st.header("1. Upload Images")
    nucleus_img = load_channel("Nucleus (Blue)", "nucleus")
    channel1_img = load_channel("Channel 1 (Red)", "ch1")
    channel2_img = load_channel("Channel 2 (Green)", "ch2")

# Processamento principal
if nucleus_img is not None:
    # Converter e processar imagem do núcleo
    nucleus_bgr = cv2.cvtColor(nucleus_img, cv2.COLOR_RGB2BGR)
    
    # Configurações na sidebar
    with st.sidebar:
        st.header("2. Nucleus Settings")
        min_distance = st.slider("Min distance (px)", 5, 50, 15, key="dist")
        min_size = st.slider("Min cell size (px)", 20, 200, 50, key="size")
        threshold = st.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.01, key="thresh")
    
    # Segmentação do núcleo
    markers = enhanced_segment_cells(nucleus_bgr, min_distance, min_size, threshold)
    
    # Visualização dos núcleos
    st.header("Nucleus Detection")
    
    # Criar visualização com bolinhas pequenas (2px)
    nucleus_vis = nucleus_img.copy()
    for i in np.unique(markers):
        if i == 0:
            continue
        mask = np.where(markers == i, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(nucleus_vis, contours, -1, (255, 0, 0), 1)  # Contorno azul fino
        
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(nucleus_vis, (cX, cY), 2, (0, 255, 255), -1)  # Bolinha amarela (2px)
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    with col1:
        st.image(nucleus_img, caption="Original Nucleus", use_column_width=True)
    with col2:
        st.image(nucleus_vis, caption="Detected Nuclei (Yellow dots)", use_column_width=True)
    
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
                st.header(f"{name} Channel Settings")
                lower = st.slider(f"Lower threshold", 0, 255, 50, key=f"{name}_lower")
                upper = st.slider(f"Upper threshold", 0, 255, 200, key=f"{name}_upper")
                dilation = st.slider(f"Dilation", 0, 5, 1, key=f"{name}_dil")
                sensitivity = st.slider(f"Sensitivity", 1, 100, 30, key=f"{name}_sens")
            
            # Processar canal
            ch_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Detecção com limiarização adaptativa
            gray = cv2.cvtColor(ch_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            _, mask = cv2.threshold(gray, lower, upper, cv2.THRESH_BINARY)
            
            # Operações morfológicas
            kernel = np.ones((3,3), np.uint8)
            if dilation > 0:
                mask = cv2.dilate(mask, kernel, iterations=dilation)
            
            # Visualização
            ch_vis = img.copy()
            positive_cells = 0
            
            # Verificar cada célula
            for i in np.unique(markers):
                if i == 0:
                    continue
                
                cell_mask = np.where(markers == i, 255, 0).astype(np.uint8)
                overlap = cv2.bitwise_and(cell_mask, mask)
                
                # Se a sobreposição for significativa
                if np.sum(overlap) > (sensitivity * 100):
                    positive_cells += 1
                    M = cv2.moments(cell_mask)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.circle(ch_vis, (cX, cY), 2, (255, 255, 0), -1)  # Bolinha ciano (2px)
            
            # Mostrar resultados
            cols = st.columns(2)
            with cols[0]:
                st.image(img, caption=f"Original {name}", use_column_width=True)
            with cols[1]:
                st.image(ch_vis, caption=f"{name} Positive Cells", use_column_width=True)
            
            results.append({
                "Channel": name,
                "Positive Cells": positive_cells,
                "Total Cells": len(np.unique(markers)) - 1,
                "Percentage": (positive_cells / (len(np.unique(markers)) - 1)) * 100 if (len(np.unique(markers)) - 1) > 0 else 0
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
