import streamlit as st
import cv2
import numpy as np
from skimage import measure
from skimage.segmentation import watershed
from scipy import ndimage
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Cell Analyzer Ultimate", layout="wide")
st.title("🔬 Ultimate Cell Analysis")

def segment_cells(image, min_distance=10, min_size=50, threshold=0.5):
    """Segmentação avançada usando Watershed"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, threshold*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    
    markers = watershed(-dist_transform, markers, mask=sure_bg)
    
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

# Sidebar para uploads
with st.sidebar:
    st.header("1. Upload Images")
    nucleus_img = load_channel("Nucleus (Blue)", "nucleus")
    channel1_img = load_channel("Channel 1 (Red)", "ch1")
    channel2_img = load_channel("Channel 2 (Green)", "ch2")

# Main content
if nucleus_img is not None:
    # Converter imagens
    nucleus_bgr = cv2.cvtColor(nucleus_img, cv2.COLOR_RGB2BGR)
    
    # Sidebar para parâmetros
    with st.sidebar:
        st.header("2. Analysis Settings")
        min_distance = st.slider("Min distance (pixels)", 5, 50, 15, key="dist")
        min_size = st.slider("Min cell size (pixels)", 20, 200, 50, key="size")
        threshold = st.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.01, key="thresh")
    
    # Segmentação do núcleo
    markers = segment_cells(nucleus_bgr, min_distance, min_size, threshold)
    
    # Visualização dos núcleos
    st.header("Nucleus Segmentation")
    vis = nucleus_img.copy()
    
    # Encontrar e desenhar centroides (bolinhas menores)
    for i in np.unique(markers):
        if i == 0:
            continue
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers == i] = 255
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(vis, (cX, cY), 1, (0, 255, 255), -1)  # Bolinha menor (raio 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(nucleus_img, caption="Original Nucleus", use_column_width=True)
    with col2:
        st.image(vis, caption="Segmented Nuclei (Yellow dots = centroids)", use_column_width=True)
    
    # Análise dos canais
    if channel1_img is not None or channel2_img is not None:
        st.header("Channel Analysis")
        
        # Processar cada canal
        results = []
        channel_names = []
        channel_images = []
        
        if channel1_img is not None:
            channel_names.append("Channel 1 (Red)")
            channel_images.append(channel1_img)
        
        if channel2_img is not None:
            channel_names.append("Channel 2 (Green)")
            channel_images.append(channel2_img)
        
        for idx, (name, img) in enumerate(zip(channel_names, channel_images)):
            # Sidebar para configurações do canal
            with st.sidebar:
                st.subheader(f"{name} Settings")
                lower = st.slider(f"Lower threshold", 0, 255, 50, key=f"{name}_lower")
                upper = st.slider(f"Upper threshold", 0, 255, 200, key=f"{name}_upper")
                dilation = st.slider(f"Dilation", 0, 5, 1, key=f"{name}_dil")
            
            # Processamento do canal
            ch_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mask = cv2.inRange(ch_bgr, (lower, lower, lower), (upper, upper, upper))
            
            kernel = np.ones((3,3), np.uint8)
            if dilation > 0:
                mask = cv2.dilate(mask, kernel, iterations=dilation)
            
            # Visualização
            ch_vis = img.copy()
            positive_cells = 0
            
            for i in np.unique(markers):
                if i == 0:
                    continue
                
                cell_mask = np.zeros_like(markers, dtype=np.uint8)
                cell_mask[markers == i] = 255
                
                overlap = cv2.bitwise_and(cell_mask, mask)
                if np.sum(overlap) > 0:
                    positive_cells += 1
                    M = cv2.moments(cell_mask)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.circle(ch_vis, (cX, cY), 1, (255, 255, 0), -1)  # Bolinha ciano pequena
            
            # Mostrar resultados
            cols = st.columns(2)
            with cols[0]:
                st.image(img, caption=f"Original {name}", use_column_width=True)
            with cols[1]:
                st.image(ch_vis, caption=f"{name} Analysis", use_column_width=True)
            
            results.append({
                "Channel": name,
                "Positive Cells": positive_cells,
                "Total Cells": len(np.unique(markers)) - 1,
                "Percentage": (positive_cells / (len(np.unique(markers)) - 1)) * 100 if (len(np.unique(markers)) - 1) > 0 else 0
            })
        
        # Resultados finais
        st.header("Quantitative Results")
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
