import streamlit as st
import cv2
import numpy as np
from skimage import measure
import pandas as pd
from PIL import Image

st.title("🔬 Multi-Color Cell Quantifier")

# 1. Upload da Imagem
uploaded_file = st.file_uploader("Carregue uma imagem multicolorida (PNG/JPG/TIFF)", type=["png", "jpg", "tif"])

if uploaded_file is not None:
    img = np.array(Image.open(uploaded_file))  # Lê a imagem em RGB
    st.image(img, caption="Imagem Original", use_column_width=True)

    # 2. Definir cores de interesse (ajustáveis pelo usuário)
    st.sidebar.header("Configurações de Cores")
    colors = {
        "Vermelho": [255, 0, 0],
        "Verde": [0, 255, 0],
        "Azul": [0, 0, 255],
        "Amarelo": [255, 255, 0],
        "Magenta": [255, 0, 255]
    }
    
    selected_colors = st.sidebar.multiselect(
        "Selecione as cores para quantificar",
        list(colors.keys()),
        default=["Vermelho", "Verde"]
    )

    # 3. Parâmetros de sensibilidade
    threshold = st.sidebar.slider("Limiar de intensidade", 0, 255, 50)
    color_tolerance = st.sidebar.slider("Tolerância de cor", 0, 100, 20)

    # 4. Processamento para cada cor selecionada
    results = []
    for color_name in selected_colors:
        target_color = np.array(colors[color_name])
        
        # Criar máscara de cor com tolerância
        lower_bound = target_color - color_tolerance
        upper_bound = target_color + color_tolerance
        mask = cv2.inRange(img, lower_bound, upper_bound)
        
        # Contar células na máscara
        labels = measure.label(mask)
        props = measure.regionprops(labels)
        cell_count = len(props)
        
        # Salvar resultados
        results.append({
            "Cor": color_name,
            "Células Detectadas": cell_count,
            "Máscara": mask
        })

    # 5. Exibir resultados
    st.header("Resultados")
    
    # Tabela de contagem
    df = pd.DataFrame(results)
    st.dataframe(df[["Cor", "Células Detectadas"]])
    
    # Visualização das máscaras
    st.subheader("Máscaras de Cores")
    cols = st.columns(len(selected_colors))
    for idx, color_name in enumerate(selected_colors):
        with cols[idx]:
            st.image(results[idx]["Máscara"], caption=f"{color_name}", use_column_width=True)

    # Exportar dados
    st.download_button(
        "Baixar Resultados (CSV)",
        df.to_csv(index=False),
        "resultados_cores.csv"
    )
