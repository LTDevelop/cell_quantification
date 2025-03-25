import streamlit as st
import cv2
import numpy as np
from skimage import measure
import pandas as pd
from PIL import Image

# Configuração do App
st.set_page_config(page_title="Cell Color Analyzer Pro", layout="wide")
st.title("🔬 Analisador Avançado de Células por Cor")

# Função para melhorar a detecção
def enhance_cell_detection(mask, min_size):
    # Operações morfológicas para limpeza
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Preenchimento de buracos
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(cleaned, [cnt], 0, 255, -1)
    
    # Filtro por tamanho
    labels = measure.label(cleaned)
    props = measure.regionprops(labels)
    
    final_mask = np.zeros_like(cleaned)
    for prop in props:
        if prop.area >= min_size:
            final_mask[labels == prop.label] = 255
            
    return final_mask

# Upload da imagem
uploaded_file = st.file_uploader("Carregue imagem de imunofluorescência", type=["png", "jpg", "tif"])

if uploaded_file is not None:
    try:
        # Processamento inicial
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Controles do usuário
        st.sidebar.header("Configurações de Análise")
        color_options = ["Vermelho", "Verde", "Azul"]
        selected_colors = st.sidebar.multiselect("Cores para análise", color_options, default=color_options)
        
        threshold = st.sidebar.slider("Sensibilidade", 0, 255, 40)
        color_tolerance = st.sidebar.slider("Tolerância de Cor", 0, 100, 30)
        min_cell_size = st.sidebar.slider("Tamanho Mínimo (pixels)", 10, 200, 50)
        enhancement = st.sidebar.checkbox("Melhorar detecção (recomendado)", value=True)

        # Definições de cores (BGR)
        color_defs = {
            "Vermelho": {"lower": [0, 0, 150], "upper": [100, 100, 255]},
            "Verde": {"lower": [0, 150, 0], "upper": [100, 255, 100]},
            "Azul": {"lower": [150, 0, 0], "upper": [255, 100, 100]}
        }

        # Processamento para cada cor
        results = []
        color_maps = []
        
        for color in selected_colors:
            # Criar máscara de cor
            lower = np.array(color_defs[color]["lower"], dtype=np.uint8)
            upper = np.array(color_defs[color]["upper"], dtype=np.uint8)
            mask = cv2.inRange(img_bgr, lower, upper)
            
            # Aplicar melhorias na detecção
            if enhancement:
                mask = enhance_cell_detection(mask, min_cell_size)
            
            # Contar células
            labels = measure.label(mask)
            props = measure.regionprops(labels)
            cells = [prop for prop in props if prop.area >= min_cell_size]
            
            # Criar imagem com apenas a cor selecionada
            color_only = np.zeros_like(img_bgr)
            color_only[mask > 0] = img_bgr[mask > 0]
            color_only_rgb = cv2.cvtColor(color_only, cv2.COLOR_BGR2RGB)
            
            # Criar overlay
            overlay = img_array.copy()
            overlay[mask > 0] = [255, 255, 255]  # Destacar células
            
            results.append({
                "Cor": color,
                "Contagem": len(cells),
                "Imagem Colorida": color_only_rgb,
                "Overlay": overlay,
                "Máscara": mask
            })
            
            # Para o mapa combinado
            color_map = np.zeros_like(img_bgr)
            color_map[mask > 0] = color_defs[color]["upper"]
            color_maps.append(color_map)

        # Mostrar resultados individuais
        st.header("Análise Individual por Cor")
        cols = st.columns(len(selected_colors))
        
        for idx, result in enumerate(results):
            with cols[idx]:
                st.image(
                    result["Imagem Colorida"],
                    caption=f"{result['Cor']}: {result['Contagem']} células",
                    use_column_width=True
                )
                st.image(
                    result["Overlay"],
                    caption=f"Células detectadas ({result['Cor']})",
                    use_column_width=True
                )

        # Resultados combinados
        st.header("Resultados Combinados")
        
        # Tabela de contagem
        df = pd.DataFrame([{"Cor": r["Cor"], "Células Detectadas": r["Contagem"]} for r in results])
        st.dataframe(df, hide_index=True)
        
        # Mapa de cores combinado
        combined_map = np.zeros_like(img_bgr)
        for cmap in color_maps:
            combined_map = cv2.add(combined_map, cmap)
        
        st.image(
            cv2.cvtColor(combined_map, cv2.COLOR_BGR2RGB),
            caption="Mapa de Cores Combinadas",
            use_column_width=True
        )

        # Exportar resultados
        st.download_button(
            "Baixar Resultados (CSV)",
            df.to_csv(index=False, encoding='utf-8-sig'),
            "contagem_celulas.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Erro durante o processamento: {str(e)}")
