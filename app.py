import streamlit as st
import cv2
import numpy as np
from skimage import measure
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Multi-Channel Cell Analyzer", layout="wide")
st.title("🔬 Multi-Channel Cell Quantification")

def process_channel(img_bgr, lower, upper, min_size, dilation=1):
    """Processa um canal de cor independentemente"""
    mask = cv2.inRange(img_bgr, lower, upper)
    
    # Operações morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    if dilation > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilation)
    
    # Identificar células
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    
    # Criar máscara final e lista de células
    final_mask = np.zeros_like(mask)
    cells = []
    for prop in props:
        if prop.area >= min_size:
            final_mask[labels == prop.label] = 255
            y, x = prop.centroid
            cells.append({
                "centroid": (int(x), int(y)),
                "area": prop.area,
                "bbox": prop.bbox
            })
            
    return final_mask, cells

# Upload da imagem
uploaded_file = st.file_uploader("Upload multi-channel image", type=["png", "jpg", "tif"])

if uploaded_file is not None:
    try:
        # Carregar imagem
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Mostrar original
        st.image(img_array, caption="Original Image", use_column_width=True)

        # Canais de cores (BGR)
        channels = {
            "Blue": {
                "lower": [120, 0, 0],
                "upper": [255, 50, 50],
                "color": [255, 0, 0]  # Exibido como vermelho
            },
            "Green": {
                "lower": [0, 120, 0],
                "upper": [50, 255, 50],
                "color": [0, 255, 0]
            },
            "Red": {
                "lower": [0, 0, 120],
                "upper": [50, 50, 255],
                "color": [0, 0, 255]
            }
        }

        # Controles
        st.sidebar.header("Channel Settings")
        
        # Processar cada canal independentemente
        results = {}
        for channel in channels:
            settings = st.sidebar.expander(f"{channel} Channel")
            
            with settings:
                lower = settings.slider(f"Lower threshold", 0, 255, channels[channel]["lower"][0], key=f"{channel}_lower")
                upper = settings.slider(f"Upper threshold", 0, 255, channels[channel]["upper"][0], key=f"{channel}_upper")
                min_size = settings.slider(f"Min size (pixels)", 10, 200, 30, key=f"{channel}_min")
                dilation = settings.slider(f"Dilation", 0, 5, 1, key=f"{channel}_dil")
            
            # Processar canal
            mask, cells = process_channel(
                img_bgr,
                np.array([lower, channels[channel]["lower"][1], channels[channel]["lower"][2]]),
                np.array([upper, channels[channel]["upper"][1], channels[channel]["upper"][2]]),
                min_size,
                dilation
            )
            
            results[channel] = {
                "mask": mask,
                "cells": cells,
                "color": channels[channel]["color"]
            }

        # Visualização dos canais individuais
        st.header("Individual Channel Results")
        cols = st.columns(3)
        
        for idx, channel in enumerate(channels):
            # Criar imagem do canal
            channel_img = np.zeros_like(img_bgr)
            channel_img[results[channel]["mask"] > 0] = results[channel]["color"]
            
            # Adicionar marcadores
            marked_img = cv2.cvtColor(channel_img, cv2.COLOR_BGR2RGB)
            for cell in results[channel]["cells"]:
                cv2.circle(marked_img, cell["centroid"], 5, (255, 255, 255), -1)
            
            with cols[idx]:
                st.image(
                    marked_img,
                    caption=f"{channel} Channel: {len(results[channel]['cells'])} cells",
                    use_column_width=True
                )

        # Análise combinada
        st.header("Combined Analysis")
        
        # Criar imagem combinada
        combined = np.zeros_like(img_bgr)
        for channel in channels:
            combined[results[channel]["mask"] > 0] = results[channel]["color"]
        
        # Adicionar todos os marcadores
        marked_combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        for channel in channels:
            for cell in results[channel]["cells"]:
                cv2.circle(marked_combined, cell["centroid"], 3, (255, 255, 255), -1)
        
        st.image(marked_combined, caption="All Detected Cells", use_column_width=True)

        # Identificar células com múltiplas cores
        st.subheader("Multi-Channel Cells")
        
        # Criar máscaras combinadas
        blue_mask = results["Blue"]["mask"]
        red_mask = results["Red"]["mask"]
        green_mask = results["Green"]["mask"]
        
        # Combinações
        combos = {
            "Blue+Red": cv2.bitwise_and(blue_mask, red_mask),
            "Blue+Green": cv2.bitwise_and(blue_mask, green_mask),
            "Red+Green": cv2.bitwise_and(red_mask, green_mask),
            "All Channels": cv2.bitwise_and(cv2.bitwise_and(blue_mask, red_mask), green_mask)
        }
        
        # Visualizar combinações
        combo_cols = st.columns(4)
        combo_results = {}
        
        for idx, (combo_name, combo_mask) in enumerate(combos.items()):
            # Contar células na combinação
            labels = measure.label(combo_mask)
            cells = measure.regionprops(labels)
            filtered_cells = [prop for prop in cells if prop.area >= 20]
            
            # Criar visualização
            combo_display = np.zeros_like(img_bgr)
            if combo_name == "Blue+Red":
                combo_display[combo_mask > 0] = [0, 0, 255]  # Magenta
            elif combo_name == "Blue+Green":
                combo_display[combo_mask > 0] = [0, 255, 255]  # Ciano
            elif combo_name == "Red+Green":
                combo_display[combo_mask > 0] = [0, 255, 0]  # Amarelo
            else:
                combo_display[combo_mask > 0] = [255, 255, 255]  # Branco
            
            # Adicionar marcadores
            marked_combo = cv2.cvtColor(combo_display, cv2.COLOR_BGR2RGB)
            for prop in filtered_cells:
                y, x = prop.centroid
                cv2.circle(marked_combo, (int(x), int(y)), 5, (255, 0, 255), -1)
            
            with combo_cols[idx]:
                st.image(
                    marked_combo,
                    caption=f"{combo_name}: {len(filtered_cells)} cells",
                    use_column_width=True
                )
            
            combo_results[combo_name] = len(filtered_cells)

        # Resultados quantitativos
        st.header("Quantitative Results")
        
        # Total de células por canal
        total_blue = len(results["Blue"]["cells"])
        total_red = len(results["Red"]["cells"])
        total_green = len(results["Green"]["cells"])
        
        # Criar tabela
        data = {
            "Channel": ["Blue", "Red", "Green"],
            "Cell Count": [total_blue, total_red, total_green],
            "% of Total": [
                (total_blue / total_blue * 100) if total_blue > 0 else 0,
                (total_red / total_blue * 100) if total_blue > 0 else 0,
                (total_green / total_blue * 100) if total_blue > 0 else 0
            ]
        }
        
        # Adicionar combinações
        for combo_name, count in combo_results.items():
            data["Channel"].append(combo_name)
            data["Cell Count"].append(count)
            data["% of Total"].append((count / total_blue * 100) if total_blue > 0 else 0)
        
        df = pd.DataFrame(data)
        st.dataframe(df, hide_index=True)
        
        # Exportar
        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "cell_quantification.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
