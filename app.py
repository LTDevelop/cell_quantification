import streamlit as st
import cv2
import numpy as np
from skimage import measure
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Multi-Color Cell Analyzer", layout="wide")
st.title("🔬 Multi-Color Cell Quantification")

def apply_color_mask(img_bgr, lower, upper, min_size=20, dilation=1):
    """Aplica máscara de cor com pós-processamento"""
    mask = cv2.inRange(img_bgr, lower, upper)
    
    # Operações morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
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

def create_colored_mask(mask, color_bgr):
    """Cria imagem colorida a partir da máscara"""
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored[mask > 0] = color_bgr
    return colored

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
            "Blue": {
                "lower": [120, 0, 0],   # BGR - Azul
                "upper": [255, 50, 50],
                "display": [255, 0, 0]  # Vermelho para exibição (contraste)
            },
            "Red": {
                "lower": [0, 0, 120],   # BGR - Vermelho
                "upper": [50, 50, 255],
                "display": [0, 0, 255]  # Vermelho puro
            },
            "Green": {
                "lower": [0, 120, 0],  # BGR - Verde
                "upper": [50, 255, 50],
                "display": [0, 255, 0]  # Verde puro
            }
        }

        # Controles para cada cor
        st.sidebar.header("Color Detection Settings")
        
        # Configurações do Azul (Núcleo)
        blue_settings = st.sidebar.expander("🔵 Blue (Nucleus) Settings")
        with blue_settings:
            blue_lower = blue_settings.slider("Blue lower threshold", 0, 255, 120, key="blue_lower")
            blue_upper = blue_settings.slider("Blue upper threshold", 0, 255, 255, key="blue_upper")
            blue_min = blue_settings.slider("Blue min size", 10, 200, 30, key="blue_min")
            blue_dilation = blue_settings.slider("Blue dilation", 0, 5, 1, key="blue_dil")
        
        # Processar núcleo (azul)
        blue_mask = apply_color_mask(
            img_bgr,
            np.array([blue_lower, 0, 0]),
            np.array([blue_upper, 50, 50]),
            blue_min,
            blue_dilation
        )
        
        # Contar núcleos
        blue_labels = measure.label(blue_mask)
        blue_props = measure.regionprops(blue_labels)
        total_cells = len([prop for prop in blue_props if prop.area >= blue_min])

        # Processar outras cores
        results = []
        color_masks = {"Blue": blue_mask}
        
        for color_name in ["Red", "Green"]:
            settings = st.sidebar.expander(f"{'🟢' if color_name == 'Green' else '🔴'} {color_name} Settings")
            
            with settings:
                lower = settings.slider(f"{color_name} lower", 0, 255, colors[color_name]["lower"][0], key=f"{color_name}_lower")
                upper = settings.slider(f"{color_name} upper", 0, 255, colors[color_name]["upper"][0], key=f"{color_name}_upper")
                min_size = settings.slider(f"{color_name} min size", 10, 100, 20, key=f"{color_name}_min")
                dilation = settings.slider(f"{color_name} dilation", 0, 5, 1, key=f"{color_name}_dil")
                require_nucleus = settings.checkbox(f"Require nucleus overlap", value=True, key=f"{color_name}_nuc")
            
            # Criar máscara
            mask = apply_color_mask(
                img_bgr,
                np.array([colors[color_name]["lower"][0], colors[color_name]["lower"][1], colors[color_name]["lower"][2]]),
                np.array([upper, colors[color_name]["upper"][1], colors[color_name]["upper"][2]]),
                min_size,
                dilation
            )
            
            # Sobrepor com núcleo se necessário
            if require_nucleus:
                mask = cv2.bitwise_and(mask, blue_mask)
            
            # Contar células
            labels = measure.label(mask)
            props = measure.regionprops(labels)
            count = len([prop for prop in props if prop.area >= min_size])
            percentage = (count / total_cells * 100) if total_cells > 0 else 0
            
            # Armazenar resultados
            color_masks[color_name] = mask
            results.append({
                "Color": color_name,
                "Count": count,
                "Percentage": percentage,
                "Mask": mask
            })

        # Visualização
        st.header("Color Detection Results")
        
        # 1. Mostrar cada cor individualmente
        st.subheader("Individual Color Channels")
        cols = st.columns(3)
        
        # Azul (Núcleo)
        blue_display = create_colored_mask(blue_mask, colors["Blue"]["display"])
        with cols[0]:
            st.image(
                cv2.cvtColor(blue_display, cv2.COLOR_BGR2RGB),
                caption=f"Blue Nucleus ({total_cells} cells)",
                use_column_width=True
            )
        
        # Vermelho e Verde
        for idx, color_name in enumerate(["Red", "Green"], 1):
            if color_name in color_masks:
                color_display = create_colored_mask(color_masks[color_name], colors[color_name]["display"])
                with cols[idx]:
                    st.image(
                        cv2.cvtColor(color_display, cv2.COLOR_BGR2RGB),
                        caption=f"{color_name}: {results[idx-1]['Count']} cells ({results[idx-1]['Percentage']:.1f}%)",
                        use_column_width=True
                    )

        # 2. Mostrar combinações de cores
        st.subheader("Color Combinations")
        
        # Criar combinações
        combo_masks = {
            "Blue+Red": cv2.bitwise_and(color_masks["Blue"], color_masks["Red"]),
            "Blue+Green": cv2.bitwise_and(color_masks["Blue"], color_masks["Green"]),
            "Red+Green": cv2.bitwise_and(color_masks["Red"], color_masks["Green"]),
            "All Colors": cv2.bitwise_and(cv2.bitwise_and(color_masks["Blue"], color_masks["Red"]), color_masks["Green"])
        }
        
        # Cores para exibição das combinações
        combo_colors = {
            "Blue+Red": [0, 0, 255],    # Vermelho
            "Blue+Green": [0, 255, 0],   # Verde
            "Red+Green": [0, 255, 255],  # Amarelo
            "All Colors": [255, 255, 255] # Branco
        }
        
        # Mostrar combinações
        combo_cols = st.columns(4)
        combo_results = []
        
        for idx, (combo_name, combo_mask) in enumerate(combo_masks.items()):
            # Contar células na combinação
            labels = measure.label(combo_mask)
            props = measure.regionprops(labels)
            combo_count = len([prop for prop in props if prop.area >= 20])
            
            # Criar visualização
            combo_display = create_colored_mask(combo_mask, combo_colors[combo_name])
            
            with combo_cols[idx % 4]:
                st.image(
                    cv2.cvtColor(combo_display, cv2.COLOR_BGR2RGB),
                    caption=f"{combo_name}: {combo_count} cells",
                    use_column_width=True
                )
            
            combo_results.append({
                "Combination": combo_name,
                "Count": combo_count,
                "Percentage": (combo_count / total_cells * 100) if total_cells > 0 else 0
            })

        # 3. Mostrar overlay completo
        st.subheader("Complete Overlay")
        
        # Criar overlay combinando todas as cores
        overlay = np.zeros_like(img_bgr)
        
        # Adicionar cada cor ao overlay
        overlay[color_masks["Blue"] > 0] = colors["Blue"]["display"]
        if "Red" in color_masks:
            overlay[color_masks["Red"] > 0] = colors["Red"]["display"]
        if "Green" in color_masks:
            overlay[color_masks["Green"] > 0] = colors["Green"]["display"]
        
        # Adicionar combinações (sobrescrevendo as cores individuais)
        for combo_name, combo_mask in combo_masks.items():
            overlay[combo_mask > 0] = combo_colors[combo_name]
        
        st.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            caption="Complete Color Overlay",
            use_column_width=True
        )

        # 4. Resultados quantitativos
        st.header("Quantitative Results")
        
        # Preparar dados para tabela
        table_data = []
        
        # Cores individuais
        for result in results:
            table_data.append({
                "Population": result["Color"],
                "Cell Count": result["Count"],
                "Percentage": result["Percentage"]
            })
        
        # Combinações
        for combo in combo_results:
            table_data.append({
                "Population": combo["Combination"],
                "Cell Count": combo["Count"],
                "Percentage": combo["Percentage"]
            })
        
        # Adicionar total de núcleos
        table_data.insert(0, {
            "Population": "Total Nuclei (Blue)",
            "Cell Count": total_cells,
            "Percentage": 100.0
        })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, hide_index=True)
        
        # Exportar dados
        st.download_button(
            "Download Results (CSV)",
            df.to_csv(index=False),
            "cell_quantification_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
