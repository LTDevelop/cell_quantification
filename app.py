import streamlit as st
import cv2
import numpy as np
from skimage import measure

st.title("🔬 Cell Quantification Tool")

uploaded_file = st.file_uploader("Upload an immunofluorescence image", type=["png", "jpg", "tif"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    threshold = st.slider("Intensity threshold", 0, 255, 50)
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    labels = measure.label(binary_img)
    props = measure.regionprops(labels, intensity_image=img)
    positive_cells = sum(1 for prop in props if prop.mean_intensity > threshold)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_column_width=True)
    with col2:
        st.image(binary_img, caption="Detected Cells", use_column_width=True)
    
    st.success(f"**Positive cells:** {positive_cells} / {len(props)}")
