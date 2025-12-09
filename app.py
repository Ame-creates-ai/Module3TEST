import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageColor
import io

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def convert_image_to_bytes(img_array):
    """Converts a numpy array (OpenCV) to PNG bytes for download."""
    # Check if image has Alpha channel (4 channels) or just 3.
    if img_array.shape[2] == 4: 
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA)
    else:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def process_signature(image_pil, threshold_val, tint_color_hex, tint_intensity):
    """
    Logic from application_e_signature.py:
    Grayscale -> Threshold -> Tint -> Merge Alpha
    """
    # 1. Read and Convert
    img_np = np.array(image_pil)
    # Handle if image is already RGBA or Grayscale, force to BGR
    if len(img_np.shape) == 2:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 2. Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Create Alpha Mask (Thresholding)
    # Invert so ink is white (255) and paper is black (0)
    _, alpha_mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # 4. Enhance Color (Tinting)
    c_rgb = ImageColor.getcolor(tint_color_hex, "RGB")
    c_bgr = (c_rgb[2], c_rgb[1], c_rgb[0]) # Flip to BGR
    
    color_mask = np.full_like(img_bgr, c_bgr)
    
    # Blend original ink with color mask
    sig_color = cv2.addWeighted(img_bgr, 1 - tint_intensity, color_mask, tint_intensity, 0)
    
    # 5. Merge Channels
    b, g, r = cv2.split(sig_color)
    final_png = cv2.merge([b, g, r, alpha_mask])
    
    return final_png, alpha_mask

def remove_background(image_pil, threshold_val):
    """
    Logic from Alpha_Channel.ipynb:
    Grayscale -> Threshold -> Alpha Channel (No Tinting)
    """
    img_np = np.array(image_pil)
    if img_np.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Create Mask
    _, alpha_mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Merge
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, alpha_mask])

# ==========================================
# MAIN APP LAYOUT
# ==========================================

st.set_page_config(page_title="Image Processing Studio", layout="wide")
st.title("🎨 Image Processing Studio")
st.markdown("This application combines techniques from **Thresholding**, **Logical Operations**, and **Alpha Channels**.")

# Create 3 Tabs
tab1, tab2, tab3 = st.tabs(["✒️ E-Signature", "📄 Watermark", "✂️ Background Remover"])

# --- TAB 1: E-SIGNATURE ---
with tab1:
    st.header("1. Create Digital Signature")
    st.info("Extracts ink from paper, tints it, and makes it transparent.")
    
    sig_file = st.file_uploader("Upload Signature", type=['jpg', 'png', 'jpeg'], key="sig")
    
    if sig_file:
        original = Image.open(sig_file)
        
        c1, c2 = st.columns(2)
        with c1:
            st.image(original, caption="Original", use_container_width=True)
            st.markdown("### Settings")
            thresh = st.slider("Threshold (Sensitivity)", 0, 255, 150, key="sig_thresh")
            color = st.color_picker("Ink Color", "#0000FF", key="sig_color")
            intensity = st.slider("Tint Intensity", 0.0, 1.0, 1.0, key="sig_int")
            
        # Process
        result_bgra, mask = process_signature(original, thresh, color, intensity)
        
        with c2:
            st.image(cv2.cvtColor(result_bgra, cv2.COLOR_BGRA2RGBA), caption="Result", use_container_width=True)
            st.download_button("Download Signature", convert_image_to_bytes(result_bgra), "signature.png", "image/png")
            with st.expander("View Mask"):
                st.image(mask, clamp=True)

# --- TAB 2: WATERMARK ---
with tab2:
    st.header("2. Watermark Document")
    st.info("Overlays a transparent image (like your signature) onto a document.")
    
    c_main, c_ctrl = st.columns([2, 1])
    with c_ctrl:
        main_doc = st.file_uploader("Document", type=['jpg', 'png'], key="wm_doc")
        wm_logo = st.file_uploader("Watermark", type=['png'], key="wm_img")
        
        if main_doc and wm_logo:
            scale = st.slider("Size", 0.1, 2.0, 0.3)
            opacity = st.slider("Opacity", 0.0, 1.0, 0.8)
            x_pos = st.slider("X Position %", 0, 100, 80)
            y_pos = st.slider("Y Position %", 0, 100, 80)

    with c_main:
        if main_doc and wm_logo:
            base = Image.open(main_doc).convert("RGBA")
            watermark = Image.open(wm_logo).convert("RGBA")
            
            # Resize
            w, h = watermark.size
            watermark = watermark.resize((int(w * scale), int(h * scale)))
            
            # Opacity
            r, g, b, a = watermark.split()
            a = a.point(lambda p: int(p * opacity))
            watermark = Image.merge('RGBA', (r, g, b, a))
            
            # Position
            bw, bh = base.size
            x = int((bw - watermark.size[0]) * (x_pos/100))
            y = int((bh - watermark.size[1]) * (y_pos/100))
            
            # Composite
            final = Image.new('RGBA', base.size)
            final = Image.alpha_composite(base, final)
            base.paste(watermark, (x, y), watermark)
            
            st.image(base, caption="Watermarked Document", use_container_width=True)
            
            # Save
            buf = io.BytesIO()
            base.save(buf, format="PNG")
            st.download_button("Download Result", buf.getvalue(), "watermarked.png", "image/png")

# --- TAB 3: BACKGROUND REMOVER ---
with tab3:
    st.header("3. Background Remover")
    st.info("Removes the background from any image (creates Alpha Channel).")
    
    bg_file = st.file_uploader("Upload Image", type=['jpg', 'png'], key="bg_rem")
    
    if bg_file:
        bg_orig = Image.open(bg_file)
        
        bc1, bc2 = st.columns(2)
        with bc1:
            st.image(bg_orig, caption="Original", use_container_width=True)
            bg_thresh = st.slider("Threshold", 0, 255, 150, key="bg_thresh")
            
        result_bg_removed = remove_background(bg_orig, bg_thresh)
        
        with bc2:
            st.image(cv2.cvtColor(result_bg_removed, cv2.COLOR_BGRA2RGBA), caption="Transparent Result", use_container_width=True)
            st.download_button("Download PNG", convert_image_to_bytes(result_bg_removed), "transparent.png", "image/png")
