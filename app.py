import streamlit as st
import cv2
import numpy as np
from skimage import measure
import pandas as pd
from PIL import Image
import sys

# Configuração da página
st.set_page_config(page_title="Cell Quantifier", layout="wide")

# Verificador de versões (debug)
def show_package_versions():
    st.sidebar.write("**Versões dos pacotes:**")
    st.sidebar.write(f"- Python: {sys.version.split()[0]}")
    st.sidebar.write(f"- OpenCV: {cv2.__version__}")
    st.sidebar.write(f"- Numpy: {np.__version__}")

show_package_versions()

# Título do app
st.title("🔬 Quantificador de Células Multicoloridas")

# 1. Upload da Imagem
uploaded_file = st.file_uploader("Carregue uma imagem de imunofluorescência", type=["png", "jpg", "tif", "tiff"])

if uploaded_file is not None:
    try:
        # 2. Carregar imagem
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        
        # Conversão para BGR (OpenCV)
        if img_array.shape[2] == 3:  # Se for RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif img_array.shape[2] == 4:  # Se for RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        st.image(img_array, caption="Imagem Original (BGR)", use_column_width=True, channels="BGR")

        # 3. Configurações de Cores
        st.sidebar.header("Configurações")
        colors = {
            "Vermelho": [0, 0, 255],  # BGR!
            "Verde": [0, 255, 0],
            "Azul": [255, 0, 0],
            "Amarelo": [0, 255, 255],
            "Magenta": [255, 0, 255]
        }
        
        selected_colors = st.sidebar.multiselect(
            "Cores para quantificar",
            list(colors.keys()),
            default=["Vermelho", "Verde"]
        )

        color_tolerance = st.sidebar.slider("Tolerância de Cor", 0, 100, 20)
        min_cell_size = st.sidebar.slider("Tamanho mínimo da célula (pixels)", 10, 100, 30)

        # 4. Processamento para cada cor
        results = []
        for color_name in selected_colors:
            target_color = np.array(colors[color_name], dtype=np.uint8)
            
            # Definir limites de cor com tolerância
            lower_bound = np.clip(target_color - color_tolerance, 0, 255).astype(np.uint8)
            upper_bound = np.clip(target_color + color_tolerance, 0, 255).astype(np.uint8)
            
            # Criar máscara
            mask = cv2.inRange(img_array, lower_bound, upper_bound)
            
            # Remover pequenos ruídos
            kernel = np.ones((3,3), np.uint8)
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Contar células
            labels = measure.label(cleaned_mask)
            props = measure.regionprops(labels)
            
            # Filtrar por tamanho
            cells = [prop for prop in props if prop.area >= min_cell_size]
            
            results.append({
                "Cor": color_name,
                "Células": len(cells),
                "Máscara": cleaned_mask
            })

        # 5. Mostrar resultados
        st.header("Resultados")
        
        # Tabela
        df = pd.DataFrame(results)
        st.dataframe(df[["Cor", "Células"]], hide_index=True)
        
        # Visualização
        st.subheader("Máscaras Geradas")
        cols = st.columns(len(selected_colors))
        for idx, color_name in enumerate(selected_colors):
            with cols[idx]:
                st.image(
                    results[idx]["Máscara"], 
                    caption=f"{color_name}: {results[idx]['Células']} células",
                    use_column_width=True
                )

        # Exportar
        st.download_button(
            "Baixar Resultados (CSV)",
            df.to_csv(index=False),
            "resultados_celulas.csv"
        )

    except Exception as e:
        st.error(f"Erro ao processar a imagem: {str(e)}")
        st.stop()
