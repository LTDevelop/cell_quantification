import streamlit as st
import cv2
import numpy as np
from skimage import measure
import pandas as pd
from PIL import Image

# Configurações do App
st.set_page_config(page_title="Cell Color Quantifier", layout="wide")
st.title("🔬 Quantificador de Células Multicoloridas")

# 1. Upload da Imagem
uploaded_file = st.file_uploader("Carregue uma imagem de imunofluorescência (RGB)", type=["png", "jpg", "tif"])

if uploaded_file is not None:
    # 2. Processamento Inicial
    img = np.array(Image.open(uploaded_file))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Converte para BGR (OpenCV)
    
    # 3. Controles Interativos
    st.sidebar.header("Configurações")
    color_options = ["Vermelho", "Verde", "Azul"]
    selected_colors = st.sidebar.multiselect("Cores para analisar", color_options, default=color_options)
    
    threshold = st.sidebar.slider("Limiar de intensidade", 0, 255, 50)
    min_cell_size = st.sidebar.slider("Tamanho mínimo da célula (pixels)", 10, 200, 30)

    # 4. Dicionário de Cores (BGR)
    color_dict = {
        "Vermelho": {'lower': [0, 0, 200], 'upper': [100, 100, 255]},
        "Verde": {'lower': [0, 200, 0], 'upper': [100, 255, 100]},
        "Azul": {'lower': [200, 0, 0], 'upper': [255, 100, 100]}
    }

    # 5. Criar máscaras para cada cor
    masks = {}
    for color in selected_colors:
        lower = np.array(color_dict[color]['lower'], dtype=np.uint8)
        upper = np.array(color_dict[color]['upper'], dtype=np.uint8)
        masks[color] = cv2.inRange(img_bgr, lower, upper)

    # 6. Identificar combinações de cores
    combinations = {
        "Vermelho": masks.get("Vermelho", np.zeros_like(masks[selected_colors[0]])),
        "Verde": masks.get("Verde", np.zeros_like(masks[selected_colors[0]])),
        "Azul": masks.get("Azul", np.zeros_like(masks[selected_colors[0]]))
    }

    # Combinações:
    combinations["Vermelho+Verde"] = cv2.bitwise_and(combinations["Vermelho"], combinations["Verde"])
    combinations["Vermelho+Azul"] = cv2.bitwise_and(combinations["Vermelho"], combinations["Azul"])
    combinations["Verde+Azul"] = cv2.bitwise_and(combinations["Verde"], combinations["Azul"])
    combinations["Todas"] = cv2.bitwise_and(combinations["Vermelho+Verde"], combinations["Azul"])

    # 7. Contagem de células por combinação
    results = []
    for name, mask in combinations.items():
        if name in selected_colors or any(c in name for c in selected_colors):
            labels = measure.label(mask)
            props = measure.regionprops(labels)
            cells = [prop for prop in props if prop.area >= min_cell_size]
            
            # Criar imagem de overlay
            overlay = img.copy()
            overlay[mask > 0] = [255, 255, 255]  # Destaca as células
            
            results.append({
                "Combinação": name,
                "Contagem": len(cells),
                "Máscara": mask,
                "Overlay": overlay
            })

    # 8. Exibir Resultados
    st.header("Resultados por Combinação de Cores")
    
    # Tabela
    df = pd.DataFrame([{'Combinação': r['Combinação'], 'Células': r['Contagem']} for r in results])
    st.dataframe(df, hide_index=True)

    # Visualização
    st.subheader("Visualização das Combinações")
    cols = st.columns(3)
    for idx, result in enumerate(results):
        with cols[idx % 3]:
            st.image(
                result['Overlay'],
                caption=f"{result['Combinação']}: {result['Contagem']} células",
                use_column_width=True
            )
            if (idx + 1) % 3 == 0 and (idx + 1) < len(results):
                cols = st.columns(3)  # Nova linha após 3 imagens

    # 9. Exportar dados
    st.download_button(
        "Baixar Resultados (CSV)",
        df.to_csv(index=False, encoding='utf-8-sig'),
        "contagem_celulas.csv",
        mime="text/csv"
    )

    # 10. Visualização Avançada (opcional)
    st.subheader("Mapa de Cores Combinadas")
    color_map = np.zeros_like(img_bgr)
    
    # Atribui cores às combinações
    color_map[combinations["Vermelho"] > 0] = [0, 0, 255]  # Vermelho
    color_map[combinations["Verde"] > 0] = [0, 255, 0]     # Verde
    color_map[combinations["Azul"] > 0] = [255, 0, 0]      # Azul
    color_map[combinations["Vermelho+Verde"] > 0] = [0, 255, 255]  # Amarelo
    color_map[combinations["Vermelho+Azul"] > 0] = [255, 0, 255]   # Magenta
    color_map[combinations["Verde+Azul"] > 0] = [255, 255, 0]      # Ciano
    color_map[combinations["Todas"] > 0] = [255, 255, 255]         # Branco

    st.image(color_map, caption="Legenda: Vermelho, Verde, Azul, Amarelo, Magenta, Ciano, Branco", use_column_width=True, channels="BGR")
