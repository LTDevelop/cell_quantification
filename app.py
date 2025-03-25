import streamlit as st
import cv2
import numpy as np
from skimage import measure
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Cell Analyzer Pro", layout="wide")
st.title("🔬 Advanced Cell Quantification")

def process_color(img_bgr, lower, upper, min_size, dilation=0):
    """Processa uma cor específica com limiarização e limpeza"""
    mask = cv2.inRange(img_bgr, lower, upper)
    
    # Operações morfológicas
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    if dilation > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilation)
    
    # Filtro por tamanho
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    
    final_mask = np.zeros_like(mask)
    for prop in props:
        if prop.area >= min_size:
            final_mask[labels == prop.label] = 255
            
    return final_mask

# Upload da imagem
uploaded_file = st.file_uploader("Upload fluorescence image", type=["png", "jpg", "tif"])

if uploaded_file is not None:
    try:
        # Carregar e converter imagem
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Mostrar imagem original
        st.image(img_array, caption="Original Image", use_column_width=True)

        # Definições de cores (BGR)
        colors = {
            "Blue (Nucleus)": {
                "lower": [100, 0, 0],
                "upper": [255, 50, 50],
                "display": [255, 0, 0]  # BGR para exibição
            },
            "Red": {
                "lower": [0, 0, 100],
                "upper": [50, 50, 255],
                "display": [0, 0, 255]
            },
            "Green": {
                "lower": [0, 100, 0],
                "upper": [50, 255, 50],
                "display": [0, 255, 0]
            },
            "White": {
                "lower": [150, 150, 150],
                "upper": [255, 255, 255],
                "display": [255, 255, 255]
            }
        }

        # Controles para cada cor
        st.sidebar.header("Color Settings")
        
        # Processar núcleo (azul) primeiro
        blue_settings = st.sidebar.expander("🔵 Nucleus (Blue) Settings")
        with blue_settings:
            blue_lower = blue_settings.slider("Blue lower threshold", 0, 255, 100, key="blue_lower")
            blue_upper = blue_settings.slider("Blue upper threshold", 0, 255, 255, key="blue_upper")
            blue_min = blue_settings.slider("Blue min size", 10, 200, 30, key="blue_min")
        
        blue_mask = process_color(
            img_bgr,
            np.array([blue_lower, 0, 0]),
            np.array([blue_upper, 50, 50]),
            blue_min
        )
        
        # Contar núcleos
        blue_labels = measure.label(blue_mask)
        blue_props = measure.regionprops(blue_labels)
        total_cells = len([prop for prop in blue_props if prop.area >= blue_min])

        # Processar outras cores
        results = []
        for color_name in ["Red", "Green", "White"]:
            settings = st.sidebar.expander(f"{'🟢' if color_name == 'Green' else '🔴' if color_name == 'Red' else '⚪'} {color_name} Settings")
            
            with settings:
                lower = settings.slider(f"{color_name} lower", 0, 255, colors[color_name]["lower"][0], key=f"{color_name}_lower")
                upper = settings.slider(f"{color_name} upper", 0, 255, colors[color_name]["upper"][0], key=f"{color_name}_upper")
                min_size = settings.slider(f"{color_name} min size", 10, 100, 20, key=f"{color_name}_min")
                dilation = settings.slider(f"{color_name} dilation", 0, 5, 1, key=f"{color_name}_dil")
            
            mask = process_color(
                img_bgr,
                np.array([lower if color_name == "Blue (Nucleus)" else colors[color_name]["lower"][0], 
                          colors[color_name]["lower"][1], 
                          colors[color_name]["lower"][2]]),
                np.array([upper if color_name == "Blue (Nucleus)" else colors[color_name]["upper"][0], 
                          colors[color_name]["upper"][1], 
                          colors[color_name]["upper"][2]]),
                min_size,
                dilation
            )
            
            # Sobrepor com núcleo
            if color_name != "Blue (Nucleus)":
                mask = cv2.bitwise_and(mask, blue_mask)
            
            # Contar células
            labels = measure.label(mask)
            props = measure.regionprops(labels)
            count = len([prop for prop in props if prop.area >= min_size])
            percentage = (count / total_cells * 100) if total_cells > 0 else 0
            
            # Criar visualização
            display_img = np.zeros_like(img_bgr)
            display_img[mask > 0] = colors[color_name]["display"]
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            # Overlay
            overlay = img_array.copy()
            overlay[mask > 0] = [255, 255, 0]  # Amarelo para destacar
            
            results.append({
                "Color": color_name,
                "Count": count,
                "Percentage": percentage,
                "Image": display_img,
                "Overlay": overlay
            })

        # Mostrar resultados
        st.header("Color Detection Results")
        
        # Mostrar núcleo primeiro
        blue_display = np.zeros_like(img_bgr)
        blue_display[blue_mask > 0] = colors["Blue (Nucleus)"]["display"]
        blue_display = cv2.cvtColor(blue_display, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(blue_display, caption=f"Blue Nucleus ({total_cells} cells)", use_column_width=True)
        with col2:
            blue_overlay = img_array.copy()
            blue_overlay[blue_mask > 0] = [255, 255, 0]
            st.image(blue_overlay, caption="Nucleus Overlay", use_column_width=True)
        
        # Mostrar outras cores
        for result in results:
            st.subheader(f"{result['Color']} Detection")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(result["Image"], 
                        caption=f"{result['Color']}: {result['Count']} cells ({result['Percentage']:.1f}%)", 
                        use_column_width=True)
            
            with col2:
                st.image(result["Overlay"], 
                        caption=f"{result['Color']} Overlay", 
                        use_column_width=True)

        # Combinações de cores
        st.header("Color Combinations")
        
        if len(results) >= 2:
            # Red + Green = Yellow
            red_mask = process_color(img_bgr, 
                                   np.array(colors["Red"]["lower"]), 
                                   np.array(colors["Red"]["upper"]), 
                                   10)
            green_mask = process_color(img_bgr, 
                                     np.array(colors["Green"]["lower"]), 
                                     np.array(colors["Green"]["upper"]), 
                                     10)
            
            yellow_mask = cv2.bitwise_and(red_mask, green_mask)
            yellow_mask = cv2.bitwise_and(yellow_mask, blue_mask)
            
            yellow_display = np.zeros_like(img_bgr)
            yellow_display[yellow_mask > 0] = [0, 255, 255]  # Amarelo em BGR
            yellow_display = cv2.cvtColor(yellow_display, cv2.COLOR_BGR2RGB)
            
            yellow_labels = measure.label(yellow_mask)
            yellow_count = len(measure.regionprops(yellow_labels))
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(yellow_display, caption=f"Red+Green (Yellow): {yellow_count} cells", use_column_width=True)
            with col2:
                yellow_overlay = img_array.copy()
                yellow_overlay[yellow_mask > 0] = [255, 0, 255]  # Magenta para destacar
                st.image(yellow_overlay, caption="Yellow Cells Overlay", use_column_width=True)

        # Resultados quantitativos
        st.header("Quantitative Analysis")
        
        data = {
            "Population": ["Total Nuclei", "Red", "Green", "White", "Red+Green"],
            "Count": [total_cells, 
                     results[0]["Count"] if len(results) > 0 else 0,
                     results[1]["Count"] if len(results) > 1 else 0,
                     results[2]["Count"] if len(results) > 2 else 0,
                     yellow_count if len(results) >= 2 else 0],
            "Percentage": [100.0,
                          results[0]["Percentage"] if len(results) > 0 else 0,
                          results[1]["Percentage"] if len(results) > 1 else 0,
                          results[2]["Percentage"] if len(results) > 2 else 0,
                          (yellow_count / total_cells * 100) if total_cells > 0 and len(results) >= 2 else 0]
        }
        
        df = pd.DataFrame(data)
        st.dataframe(df, hide_index=True)
        
        # Exportar
        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "cell_counts.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
