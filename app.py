import streamlit as st
import cv2
import numpy as np
from skimage import measure
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Advanced Cell Analyzer", layout="wide")
st.title("🔬 Advanced Cell Quantification with Markers")

def process_cells(img_bgr, lower, upper, min_size, dilation=1):
    """Processa detecção de células com limpeza morfológica"""
    mask = cv2.inRange(img_bgr, lower, upper)
    
    # Operações morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    if dilation > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilation)
    
    # Filtro por tamanho e preenchimento
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    
    final_mask = np.zeros_like(mask)
    centroids = []
    for prop in props:
        if prop.area >= min_size:
            final_mask[labels == prop.label] = 255
            centroids.append(prop.centroid)
            
    return final_mask, centroids

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
                "lower": [120, 0, 0],
                "upper": [255, 50, 50],
                "display": (255, 0, 0)  # Vermelho para visualização
            },
            "Red": {
                "lower": [0, 0, 120],
                "upper": [50, 50, 255],
                "display": (0, 0, 255)  # Vermelho
            },
            "Green": {
                "lower": [0, 120, 0],
                "upper": [50, 255, 50],
                "display": (0, 255, 0)   # Verde
            }
        }

        # Controles
        st.sidebar.header("Analysis Settings")
        
        # Configurações do núcleo (Azul)
        blue_settings = st.sidebar.expander("🔵 Nucleus Settings")
        with blue_settings:
            blue_lower = blue_settings.slider("Lower threshold", 0, 255, 120, key="blue_lower")
            blue_upper = blue_settings.slider("Upper threshold", 0, 255, 255, key="blue_upper")
            blue_min = blue_settings.slider("Min size (pixels)", 10, 200, 30, key="blue_min")
            blue_dilation = blue_settings.slider("Dilation", 0, 5, 1, key="blue_dil")
        
        # Processar núcleo
        blue_mask, blue_centroids = process_cells(
            img_bgr,
            np.array([blue_lower, 0, 0]),
            np.array([blue_upper, 50, 50]),
            blue_min,
            blue_dilation
        )
        total_cells = len(blue_centroids)

        # Processar outras cores
        results = []
        color_masks = {"Blue (Nucleus)": blue_mask}
        
        for color_name in ["Red", "Green"]:
            settings = st.sidebar.expander(f"{'🟢' if color_name == 'Green' else '🔴'} {color_name} Settings")
            
            with settings:
                lower = settings.slider("Lower threshold", 0, 255, colors[color_name]["lower"][0], key=f"{color_name}_lower")
                upper = settings.slider("Upper threshold", 0, 255, colors[color_name]["upper"][0], key=f"{color_name}_upper")
                min_size = settings.slider("Min size (pixels)", 10, 100, 20, key=f"{color_name}_min")
                dilation = settings.slider("Dilation", 0, 5, 1, key=f"{color_name}_dil")
            
            # Processar cor
            mask, centroids = process_cells(
                img_bgr,
                np.array([lower, colors[color_name]["lower"][1], colors[color_name]["lower"][2]]),
                np.array([upper, colors[color_name]["upper"][1], colors[color_name]["upper"][2]]),
                min_size,
                dilation
            )
            
            # Sobrepor com núcleo
            mask = cv2.bitwise_and(mask, blue_mask)
            
            # Contar células
            count = len(centroids)
            percentage = (count / total_cells * 100) if total_cells > 0 else 0
            
            color_masks[color_name] = mask
            results.append({
                "Color": color_name,
                "Count": count,
                "Percentage": percentage,
                "Mask": mask,
                "Centroids": centroids
            })

        # Visualização
        st.header("Detection Results")
        
        # 1. Mostrar cada canal com marcações
        st.subheader("Individual Channels with Cell Markers")
        cols = st.columns(3)
        
        # Núcleo (Azul)
        blue_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
        for y, x in blue_centroids:
            cv2.circle(blue_display, (int(x), int(y)), 5, (255, 0, 0), -1)  # Marcador azul
        with cols[0]:
            st.image(
                blue_display,
                caption=f"Blue Nucleus ({total_cells} cells)",
                use_column_width=True
            )
        
        # Cores individuais
        for idx, color_name in enumerate(["Red", "Green"]):
            if color_name in color_masks:
                color_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
                color_mask = color_masks[color_name]
                color_display[color_mask > 0] = colors[color_name]["display"]
                
                # Adicionar marcadores
                for y, x in results[idx]["Centroids"]:
                    cv2.circle(color_display, (int(x), int(y)), 5, (255, 255, 255), -1)  # Marcador branco
                
                with cols[idx+1]:
                    st.image(
                        color_display,
                        caption=f"{color_name}: {results[idx]['Count']} cells ({results[idx]['Percentage']:.1f}%)",
                        use_column_width=True
                    )

        # 2. Mostrar combinações
        st.subheader("Color Combinations")
        
        if len(results) >= 2:
            # Red + Green = Yellow
            red_mask = color_masks["Red"]
            green_mask = color_masks["Green"]
            yellow_mask = cv2.bitwise_and(red_mask, green_mask)
            
            # Contar células amarelas
            yellow_labels = measure.label(yellow_mask)
            yellow_props = measure.regionprops(yellow_labels)
            yellow_cells = len([prop for prop in yellow_props if prop.area >= 20])
            
            # Visualização
            combo_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
            combo_display[yellow_mask > 0] = (255, 255, 0)  # Amarelo
            
            # Adicionar marcadores
            for prop in yellow_props:
                y, x = prop.centroid
                cv2.circle(combo_display, (int(x), int(y)), 5, (255, 0, 255), -1)  # Marcador magenta
            
            st.image(
                combo_display,
                caption=f"Red+Green (Yellow): {yellow_cells} cells",
                use_column_width=True
            )

        # 3. Overlay completo com todas as cores
        st.subheader("Complete Overlay")
        
        overlay = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
        
        # Aplicar cores
        overlay[blue_mask > 0] = colors["Blue (Nucleus)"]["display"]
        if "Red" in color_masks:
            overlay[color_masks["Red"] > 0] = colors["Red"]["display"]
        if "Green" in color_masks:
            overlay[color_masks["Green"] > 0] = colors["Green"]["display"]
        
        # Adicionar marcadores para todas as células
        all_centroids = blue_centroids.copy()
        for result in results:
            all_centroids.extend(result["Centroids"])
        
        for y, x in all_centroids:
            cv2.circle(overlay, (int(x), int(y)), 3, (255, 255, 255), -1)  # Marcador branco
        
        st.image(overlay, caption="Complete Analysis", use_column_width=True)

        # 4. Resultados quantitativos
        st.header("Quantitative Results")
        
        data = {
            "Population": ["Total Nuclei", "Red", "Green", "Red+Green"],
            "Count": [total_cells,
                     results[0]["Count"] if len(results) > 0 else 0,
                     results[1]["Count"] if len(results) > 1 else 0,
                     yellow_cells if len(results) >= 2 else 0],
            "Percentage": [100.0,
                          results[0]["Percentage"] if len(results) > 0 else 0,
                          results[1]["Percentage"] if len(results) > 1 else 0,
                          (yellow_cells / total_cells * 100) if total_cells > 0 and len(results) >= 2 else 0]
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
