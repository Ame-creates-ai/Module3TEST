import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageColor
import io

# --- Helper Function to Download Images ---
def convert_image_to_bytes(img_array):
    """Converts a numpy array (OpenCV) to PNG bytes for download."""
    # OpenCV uses BGR, PIL uses RGB. We need to convert before saving via PIL.
    # Check if image has Alpha channel (4 channels) or just 3.
    if img_array.shape[2] == 4: 
        # BGRA -> RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA)
    else:
        # BGR -> RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

st.set_page_config(page_title="E-Signature & Watermark App", layout="wide")

st.title(" ✒️ E-Signature & Watermark Studio")
st.markdown("Use the tabs below to switch between creating a signature and watermarking a document.")

tab1, tab2 = st.tabs(["1. E-Signature Generator", "2. Watermark Document"])

# ==========================================
# TAB 1: E-SIGNATURE (Logic from application_e_signature.py)
# ==========================================
with tab1:
    st.header("1. Create E-Signature")
    st.markdown("This tool follows the steps from your notebook: **Grayscale -> Threshold -> Tint -> Merge Alpha**.")
    
    # Upload
    uploaded_sig = st.file_uploader("Upload Signature Photo", type=['jpg', 'jpeg', 'png'], key="sig_upload")
    
    if uploaded_sig:
        # 1. READ IMAGE
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_sig.read()), dtype=np.uint8)
        sig_org = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display Original
        col_orig, col_proc = st.columns(2)
        with col_orig:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(sig_org, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            # CONTROLS
            st.markdown("---")
            st.markdown("### Controls")
            
            # Threshold slider (Interactive version of the hardcoded 150)
            thresh_val = st.slider("Threshold Value", 0, 255, 150, help="Adjust until background is removed.")
            
            # Tint Color Picker (Interactive version of the blue mask)
            tint_color_hex = st.color_picker("Signature Ink Color", "#0000FF") # Default to Blue like notebook
        
        # --- PROCESSING STEPS FROM NOTEBOOK ---
        
        # 2. GRAYSCALE
        # notebook: sig_gray = cv2.cvtColor(sig, cv2.COLOR_BGR2GRAY)
        sig_gray = cv2.cvtColor(sig_org, cv2.COLOR_BGR2GRAY)
        
        # 3. THRESHOLD (ALPHA MASK)
        # notebook: ret, alpha_mask = cv2.threshold(sig_gray, 150, 255, cv2.THRESH_BINARY_INV)
        # We use the slider value 'thresh_val' instead of hardcoded 150
        ret, alpha_mask = cv2.threshold(sig_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # 4. ENHANCE COLOR (TINTING)
        # notebook uses a blue_mask and addWeighted.
        
        # Convert hex color to BGR tuple for OpenCV
        c_rgb = ImageColor.getcolor(tint_color_hex, "RGB")
        c_bgr = (c_rgb[2], c_rgb[1], c_rgb[0]) # Flip to BGR
        
        # notebook: blue_mask = sig.copy(); blue_mask[:, :] = (255, 0, 0)
        color_mask = sig_org.copy()
        color_mask[:, :] = c_bgr 
        
        # notebook: sig_color = cv2.addWeighted(sig, 1, blue_mask, 0.5, 0)
        # We allow user to adjust the blending intensity (alpha weight)
        tint_intensity = st.slider("Tint Intensity", 0.0, 1.0, 0.5)
        sig_color = cv2.addWeighted(sig_org, 1 - tint_intensity, color_mask, tint_intensity, 0)
        
        # 5. SPLIT AND MERGE CHANNELS
        # notebook: b, g, r = cv2.split(sig_color)
        b, g, r = cv2.split(sig_color)
        
        # notebook: new = [b, g, r, alpha_mask]
        # notebook: png = cv2.merge(new, 4)
        new_channels = [b, g, r, alpha_mask]
        final_png = cv2.merge(new_channels)
        
        # --- DISPLAY RESULT ---
        with col_proc:
            st.subheader("Processed Result")
            # Must convert BGRA (OpenCV) to RGBA (Streamlit/PIL)
            st.image(cv2.cvtColor(final_png, cv2.COLOR_BGRA2RGBA), use_container_width=True)
            
            # Download Button
            btn = st.download_button(
                label="Download Transparent Signature",
                data=convert_image_to_bytes(final_png),
                file_name="my_esignature.png",
                mime="image/png"
            )
            
            # Debug: Show the mask being used
            with st.expander("See Alpha Mask (Debug)"):
                st.image(alpha_mask, caption="Alpha Mask (White=Keep)", clamp=True)

# ==========================================
# TAB 2: WATERMARK (Standard Overlay Logic)
# ==========================================
with tab2:
    st.header("2. Watermark Document")
    
    col_main, col_controls = st.columns([2, 1])
    
    with col_controls:
        main_file = st.file_uploader("Upload Document", type=['jpg', 'png', 'jpeg'], key="wm_main")
        watermark_file = st.file_uploader("Upload Watermark (PNG)", type=['png', 'jpg'], key="wm_logo")
        
        if main_file and watermark_file:
            st.markdown("#### Settings")
            scale = st.slider("Watermark Scale", 0.1, 2.0, 0.3)
            opacity = st.slider("Opacity", 0.0, 1.0, 0.7)
            pos_x = st.slider("X Position", 0, 100, 50)
            pos_y = st.slider("Y Position", 0, 100, 50)

    with col_main:
        if main_file and watermark_file:
            # We use PIL here for easier alpha compositing
            base_img = Image.open(main_file).convert("RGBA")
            wm_img = Image.open(watermark_file).convert("RGBA")
            
            # Resize
            wm_w, wm_h = wm_img.size
            new_w = int(wm_w * scale)
            new_h = int(wm_h * scale)
            wm_img = wm_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Opacity
            r, g, b, a = wm_img.split()
            a = a.point(lambda p: int(p * opacity))
            wm_img = Image.merge('RGBA', (r, g, b, a))
            
            # Position
            bg_w, bg_h = base_img.size
            x = int((bg_w - new_w) * (pos_x / 100))
            y = int((bg_h - new_h) * (pos_y / 100))
            
            # Composite
            combined = Image.new('RGBA', base_img.size)
            combined = Image.alpha_composite(base_img, combined)
            combined.paste(wm_img, (x, y), wm_img)
            
            st.image(combined, use_container_width=True)
            
            # Download
            buf = io.BytesIO()
            combined.save(buf, format="PNG")
            st.download_button("Download Watermarked Doc", buf.getvalue(), "watermarked.png", "image/png")
