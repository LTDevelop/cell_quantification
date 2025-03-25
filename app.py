import streamlit as st
import cv2
import numpy as np
from skimage import measure
from skimage.segmentation import watershed
from scipy import ndimage
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Advanced Cell Analyzer Pro", layout="wide")
st.title("🔬 Advanced Multi-Channel Cell Analysis")

def segment_cells(image, min_distance=10, min_size=50, threshold=0.5):
    """Segmentação avançada usando Watershed"""
    # Pré-processamento
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold adaptativo
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Remover pequenos ruídos
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Área de fundo
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Transformada de distância
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, threshold*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Área desconhecida
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Rotular marcadores
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    
    # Aplicar Watershed
    markers = watershed(-dist_transform, markers, mask=sure_bg)
    
    # Filtrar por tamanho
    unique, counts = np.unique(markers, return_counts=True)
    for (i, count) in zip(unique, counts):
        if count < min_size and i != 0:  # 0 é o fundo
            markers[markers == i] = 0
    
    return markers

def load_channel(name, key_suffix):
    """Interface para upload de cada canal"""
    with st.expander(f"🖼️ {name} Channel", expanded=True):
        uploaded = st.file_uploader(f"Upload {name} image", 
                                   type=["png", "jpg", "tif"], 
                                   key=f"{name}_{key_suffix}")
        if uploaded:
            img = Image.open(uploaded)
            return np.array(img)
    return None

# Interface de upload
st.header("1. Upload Channel Images")
col1, col2, col3 = st.columns(3)

with col1:
    nucleus_img = load_channel("Nucleus (Blue)", "nucleus")

with col2:
    channel1_img = load_channel("Channel 1 (Red)", "ch1")

with col3:
    channel2_img = load_channel("Channel 2 (Green)", "ch2")

# Configurações de análise
st.header("2. Analysis Settings")
with st.expander("⚙️ Segmentation Parameters"):
    min_distance = st.slider("Minimum distance between cells (pixels)", 5, 50, 15)
    min_size = st.slider("Minimum cell size (pixels)", 20, 200, 50)
    threshold = st.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.01)

# Botão de análise
analyze_button = st.button("🔍 Analyze", type="primary")

if analyze_button and nucleus_img is not None:
    try:
        # Converter imagens
        nucleus_bgr = cv2.cvtColor(nucleus_img, cv2.COLOR_RGB2BGR)
        
        # Segmentação do núcleo
        markers = segment_cells(nucleus_bgr, min_distance, min_size, threshold)
        
        # Visualização dos núcleos segmentados
        st.header("3. Nucleus Segmentation Preview")
        
        # Criar imagem de visualização
        vis = nucleus_img.copy()
        for i in np.unique(markers):
            if i == 0:  # Fundo
                continue
            mask = np.zeros_like(markers, dtype=np.uint8)
            mask[markers == i] = 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)
            
            # Centroide
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(vis, (cX, cY), 3, (0, 255, 255), -1)
        
        st.image(vis, caption="Segmented Nuclei (Yellow dots = centroids)", use_column_width=True)
        
        # Análise dos canais
        if channel1_img is not None or channel2_img is not None:
            st.header("4. Channel Analysis")
            
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
            
            cols = st.columns(len(channel_names))
            
            for idx, (name, img) in enumerate(zip(channel_names, channel_images)):
                # Converter para BGR
                ch_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Configurações do canal
                with cols[idx]:
                    st.subheader(name)
                    
                    lower = st.slider(f"Lower threshold", 0, 255, 50, key=f"{name}_lower")
                    upper = st.slider(f"Upper threshold", 0, 255, 200, key=f"{name}_upper")
                    dilation = st.slider(f"Dilation", 0, 5, 1, key=f"{name}_dil")
                    
                    # Criar máscara
                    mask = cv2.inRange(ch_bgr, (lower, lower, lower), (upper, upper, upper))
                    
                    # Operações morfológicas
                    kernel = np.ones((3,3), np.uint8)
                    if dilation > 0:
                        mask = cv2.dilate(mask, kernel, iterations=dilation)
                    
                    # Contar células positivas
                    positive_cells = 0
                    ch_vis = img.copy()
                    
                    for i in np.unique(markers):
                        if i == 0:
                            continue
                        
                        # Criar máscara para a célula atual
                        cell_mask = np.zeros_like(markers, dtype=np.uint8)
                        cell_mask[markers == i] = 255
                        
                        # Verificar sobreposição com o canal
                        overlap = cv2.bitwise_and(cell_mask, mask)
                        if np.sum(overlap) > 0:
                            positive_cells += 1
                            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(ch_vis, contours, -1, (255, 0, 0), 2)
                    
                    # Mostrar resultados do canal
                    st.image(ch_vis, caption=f"{name} - {positive_cells} positive cells", use_column_width=True)
                    results.append({
                        "Channel": name,
                        "Positive Cells": positive_cells,
                        "Total Cells": len(np.unique(markers)) - 1,
                        "Percentage": (positive_cells / (len(np.unique(markers)) - 1)) * 100 if (len(np.unique(markers)) - 1) > 0 else 0
                    })
            
            # Resultados finais
            st.header("5. Quantitative Results")
            df = pd.DataFrame(results)
            st.dataframe(df, hide_index=True)
            
            # Exportar
            st.download_button(
                "📊 Download Results",
                df.to_csv(index=False),
                "cell_analysis_results.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")

elif analyze_button:
    st.warning("Please upload at least the nucleus image to start analysis!")
