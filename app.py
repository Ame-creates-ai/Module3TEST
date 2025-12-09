import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- Helper Functions ---

def convert_image_to_bytes(img_array, fmt='PNG'):
    """Converts a numpy array (OpenCV) or PIL image to bytes for download."""
    if isinstance(img_array, np.ndarray):
        # Convert BGR to RGB for PIL
        if img_array.shape[2] == 4: # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA)
        elif img_array.shape[2] == 3: # BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_array)
    else:
        pil_img = img_array
    
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    byte_im = buf.getvalue()
    return byte_im

def process_signature(image, threshold_val, color_hex):
    """
    Applies the logic from the Application_E_Signature.ipynb:
    1. Grayscale 
    2. Threshold (Alpha Mask)
    3. Tinting
    4. Channel Merging
    """
    # Convert PIL to OpenCV format (RGB -> BGR)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Create Alpha Mask (Thresholding)
    # Note: We use THRESH_BINARY_INV so ink becomes White (255) and paper becomes Black (0)
    # The mask determines what is KEPT.
    _, alpha_mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # 3. Enhance Color (Tinting)
    # Convert hex color to BGR tuple
    h = color_hex.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    bgr = (rgb[2], rgb[1], rgb[0])

    # Create a solid color block the same size as the image
    color_layer = np.full_like(img, bgr)

    # 4. Merge Channels
    # Split the color layer (or use original image if you prefer original ink color)
    b, g, r = cv2.split(color_layer)
    
    # Merge B, G, R with the Alpha Mask
    # Result is a BGRA image (transparent background)
    final_png = cv2.merge([b, g, r, alpha_mask])
    
    return final_png, alpha_mask

# --- Main App Interface ---

st.set_page_config(page_title="E-Signature & Watermark Tool", layout="wide")

st.title(" ✒️ E-Signature & Watermark Application")
st.markdown("""
This app applies computer vision techniques (Thresholding, Masking, Alpha Channels) to:
1. Extract a signature from a photo.
2. Watermark images with that signature.
""")

tab1, tab2 = st.tabs(["1. Create E-Signature", "2. Add Watermark"])

# ==========================================
# TAB 1: E-SIGNATURE GENERATOR
# ==========================================
with tab1:
    st.header("Create Transparent Signature")
    
    sig_file = st.file_uploader("Upload a photo of your signature", type=['jpg', 'jpeg', 'png'])

    if sig_file:
        # Load Image
        original_pil = Image.open(sig_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            st.image(original_pil, use_container_width=True)
            
            st.markdown("### Settings")
            # Threshold Slider
            thresh = st.slider(
                "Threshold Value", 
                min_value=0, 
                max_value=255, 
                value=150, 
                help="Adjust this until the background disappears and only ink remains."
            )
            
            # Color Picker
            ink_color = st.color_picker("Ink Color", "#000000")
            
        # Process
        result_bgra, mask = process_signature(original_pil, thresh, ink_color)
        
        with col2:
            st.subheader("Result (Transparent PNG)")
            # Display requires converting BGRA (OpenCV) to RGBA (PIL/Streamlit)
            st.image(cv2.cvtColor(result_bgra, cv2.COLOR_BGRA2RGBA), use_container_width=True)
            
            # Debug view of the mask
            with st.expander("View Alpha Mask (Debug)"):
                st.image(mask, caption="Alpha Mask (White=Keep, Black=Transparent)", clamp=True)
            
            # Download Button
            png_bytes = convert_image_to_bytes(result_bgra, "PNG")
            st.download_button(
                label="Download Signature",
                data=png_bytes,
                file_name="my_esignature.png",
                mime="image/png"
            )

# ==========================================
# TAB 2: WATERMARK ADDER
# ==========================================
with tab2:
    st.header("Watermark Your Documents")
    
    col_main, col_controls = st.columns([2, 1])
    
    with col_controls:
        main_file = st.file_uploader("Upload Main Image/Document", type=['jpg', 'png', 'jpeg'])
        watermark_file = st.file_uploader("Upload Watermark (PNG recommended)", type=['png', 'jpg'])
        
        if main_file and watermark_file:
            st.markdown("---")
            st.markdown("#### Watermark Controls")
            
            scale = st.slider("Scale", 0.1, 2.0, 0.3)
            opacity = st.slider("Opacity", 0.0, 1.0, 0.8)
            
            # Position controls
            st.markdown("#### Position")
            pos_x = st.slider("X Position (%)", 0, 100, 85)
            pos_y = st.slider("Y Position (%)", 0, 100, 90)

    with col_main:
        if main_file and watermark_file:
            # Using PIL here as it handles Alpha compositing much better than raw OpenCV
            base_img = Image.open(main_file).convert("RGBA")
            wm_img = Image.open(watermark_file).convert("RGBA")
            
            # 1. Resize Watermark
            wm_w, wm_h = wm_img.size
            new_w = int(wm_w * scale)
            new_h = int(wm_h * scale)
            wm_img = wm_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # 2. Adjust Opacity
            # Split channels, adjust alpha, merge back
            r, g, b, a = wm_img.split()
            # Apply opacity factor to the alpha channel
            a = a.point(lambda p: int(p * opacity))
            wm_img = Image.merge('RGBA', (r, g, b, a))
            
            # 3. Calculate Position
            base_w, base_h = base_img.size
            # Convert percentage to pixels
            x_loc = int((base_w - new_w) * (pos_x / 100))
            y_loc = int((base_h - new_h) * (pos_y / 100))
            
            # 4. Composite
            # We create a transparent layer the size of the base image
            transparent_layer = Image.new('RGBA', base_img.size, (0,0,0,0))
            transparent_layer.paste(wm_img, (x_loc, y_loc))
            
            # Alpha composite the base and the layer
            final_composite = Image.alpha_composite(base_img, transparent_layer)
            
            st.image(final_composite, caption="Watermarked Image", use_container_width=True)
            
            # Download
            final_bytes = convert_image_to_bytes(final_composite, "PNG")
            st.download_button(
                label="Download Watermarked Image",
                data=final_bytes,
                file_name="watermarked_doc.png",
                mime="image/png"
            )
        else:
            st.info("Please upload both a Main Image and a Watermark Image to begin.")
