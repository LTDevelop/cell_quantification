import streamlit as st
import cv2
import numpy as np
from skimage import measure
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Advanced Cell Analyzer", layout="wide")
st.title("🔬 Advanced Cell Quantification with Nucleus Reference")

def enhance_detection(mask, min_size):
    # Operações morfológicas melhoradas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    
    # Fechamento para preencher buracos
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Abertura para remover pequenos ruídos
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Preenchimento de buracos garantido
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(opened)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_size:
            cv2.drawContours(filled, [cnt], 0, 255, -1)
    
    return filled

# Upload da imagem
uploaded_file = st.file_uploader("Upload fluorescence image", type=["png", "jpg", "tif"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Cores disponíveis incluindo branco
    color_options = {
        "Blue (Nucleus)": {"lower": [120, 0, 0], "upper": [255, 50, 50], "color": [255, 0, 0]},
        "Red": {"lower": [0, 0, 120], "upper": [50, 50, 255], "color": [0, 0, 255]},
        "Green": {"lower": [0, 120, 0], "upper": [50, 255, 50], "color": [0, 255, 0]},
        "White": {"lower": [150, 150, 150], "upper": [255, 255, 255], "color": [255, 255, 255]}
    }
    
    # Processamento para cada cor com controles individuais
    results = []
    
    # Primeiro processamos o núcleo (Azul)
    st.sidebar.header("Nucleus (Blue) Settings")
    blue_threshold = st.sidebar.slider("Blue sensitivity", 0, 255, 60, key="blue_thresh")
    blue_min_size = st.sidebar.slider("Blue min size", 10, 200, 30, key="blue_size")
    
    blue_lower = np.array(color_options["Blue (Nucleus)"]["lower"], dtype=np.uint8)
    blue_upper = np.array(color_options["Blue (Nucleus)"]["upper"], dtype=np.uint8)
    blue_mask = cv2.inRange(img_bgr, blue_lower, blue_upper)
    blue_mask = enhance_detection(blue_mask, blue_min_size)
    
    # Contar núcleos (células totais)
    blue_labels = measure.label(blue_mask)
    blue_props = measure.regionprops(blue_labels)
    total_cells = len([prop for prop in blue_props if prop.area >= blue_min_size])
    
    # Processar outras cores
    for color_name in ["Red", "Green", "White"]:
        st.sidebar.header(f"{color_name} Settings")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            threshold = st.slider(f"{color_name} sensitivity", 0, 255, 40, key=f"{color_name}_thresh")
            min_size = st.slider(f"{color_name} min size", 10, 100, 20, key=f"{color_name}_size")
        
        with col2:
            tolerance = st.slider(f"{color_name} tolerance", 0, 100, 30, key=f"{color_name}_tol")
            dilation = st.slider(f"{color_name} dilation", 0, 10, 2, key=f"{color_name}_dil")
        
        # Ajustar limites com tolerância
        base_lower = np.array(color_options[color_name]["lower"], dtype=np.uint8)
        base_upper = np.array(color_options[color_name]["upper"], dtype=np.uint8)
        
        lower = np.clip(base_lower - tolerance, 0, 255)
        upper = np.clip(base_upper + tolerance, 0, 255)
        
        # Criar máscara
        mask = cv2.inRange(img_bgr, lower, upper)
        mask = enhance_detection(mask, min_size)
        
        # Dilatar se necessário
        if dilation > 0:
            kernel = np.ones((dilation, dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Sobrepor com núcleo para garantir células completas
        if color_name != "Blue (Nucleus)":
            mask = cv2.bitwise_and(mask, blue_mask)
        
        # Contar células positivas
        labels = measure.label(mask)
        props = measure.regionprops(labels)
        positive_cells = len([prop for prop in props if prop.area >= min_size])
        
        # Calcular porcentagem relativa ao núcleo
        percentage = (positive_cells / total_cells * 100) if total_cells > 0 else 0
        
        # Imagem colorida apenas com esta cor
        color_only = np.zeros_like(img_bgr)
        color_only[mask > 0] = color_options[color_name]["color"]
        color_only_rgb = cv2.cvtColor(color_only, cv2.COLOR_BGR2RGB)
        
        # Overlay de detecção
        overlay = img_array.copy()
        overlay[mask > 0] = [255, 255, 255]  # Branco para células detectadas
        
        results.append({
            "Color": color_name,
            "Count": positive_cells,
            "Percentage": percentage,
            "Image": color_only_rgb,
            "Overlay": overlay,
            "Mask": mask
        })
    
    # Visualização
    st.header("Color-Specific Analysis")
    
    # Mostrar cada cor com seus controles e resultados
    for result in results:
        st.subheader(f"{result['Color']} Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(
                result["Image"],
                caption=f"{result['Color']} only ({result['Count']} cells, {result['Percentage']:.1f}%)",
                use_column_width=True
            )
        
        with col2:
            st.image(
                result["Overlay"],
                caption=f"Detected cells (white)",
                use_column_width=True
            )
    
    # Resultados combinados
    st.header("Combined Results")
    
    # Cálculo de combinações de cores
    red_mask = results[0]["Mask"] if len(results) > 0 else np.zeros_like(blue_mask)
    green_mask = results[1]["Mask"] if len(results) > 1 else np.zeros_like(blue_mask)
    white_mask = results[2]["Mask"] if len(results) > 2 else np.zeros_like(blue_mask)
    
    # Combinações
    yellow_mask = cv2.bitwise_and(red_mask, green_mask)
    magenta_mask = cv2.bitwise_and(red_mask, blue_mask)
    cyan_mask = cv2.bitwise_and(green_mask, blue_mask)
    
    # Contar combinações
    combo_results = []
    combo_colors = {
        "Yellow (R+G)": {"mask": yellow_mask, "color": [0, 255, 255]},
        "Magenta (R+B)": {"mask": magenta_mask, "color": [255, 0, 255]},
        "Cyan (G+B)": {"mask": cyan_mask, "color": [255, 255, 0]}
    }
    
    for name, data in combo_colors.items():
        labels = measure.label(data["mask"])
        props = measure.regionprops(labels)
        count = len([prop for prop in props if prop.area >= 20])  # Tamanho mínimo para combinações
        
        if count > 0:
            combo_img = np.zeros_like(img_bgr)
            combo_img[data["mask"] > 0] = data["color"]
            combo_img_rgb = cv2.cvtColor(combo_img, cv2.COLOR_BGR2RGB)
            
            combo_results.append({
                "Combination": name,
                "Count": count,
                "Image": combo_img_rgb
            })
    
    # Mostrar combinações
    if combo_results:
        st.subheader("Color Combinations")
        cols = st.columns(len(combo_results))
        
        for idx, combo in enumerate(combo_results):
            with cols[idx]:
                st.image(
                    combo["Image"],
                    caption=f"{combo['Combination']}: {combo['Count']} cells",
                    use_column_width=True
                )
    
    # Tabela de resultados
    st.subheader("Quantitative Results")
    
    result_data = {
        "Population": ["Total Nuclei"] + [r["Color"] for r in results],
        "Cell Count": [total_cells] + [r["Count"] for r in results],
        "Percentage": [100.0] + [r["Percentage"] for r in results]
    }
    
    df = pd.DataFrame(result_data)
    st.dataframe(df, hide_index=True)
    
    # Exportar dados
    st.download_button(
        "Download Results (CSV)",
        df.to_csv(index=False),
        "cell_quantification_results.csv",
        mime="text/csv"
    )
