import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageColor
import io

# ==========================================
# I did use GPT to help me create the UI to make it nicer, and the whole code more organized. I completed all the notebooks from Module 3 as well. 
# ==========================================

def convert_image_to_bytes(img_array):
    """Converts a numpy array (OpenCV) to PNG bytes for download."""
    if img_array.shape[2] == 4: 
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA)
    else:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def process_signature(image_pil, threshold_val, tint_color_hex, tint_intensity):
    """ E-Signature Logic: Grayscale -> Threshold -> Tint -> Merge Alpha """
    img_np = np.array(image_pil)
    # Ensure BGR
    if len(img_np.shape) == 2: img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4: img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else: img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Thresholding
    _, alpha_mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # Tinting
    c_rgb = ImageColor.getcolor(tint_color_hex, "RGB")
    c_bgr = (c_rgb[2], c_rgb[1], c_rgb[0])
    color_mask = np.full_like(img_bgr, c_bgr)
    sig_color = cv2.addWeighted(img_bgr, 1 - tint_intensity, color_mask, tint_intensity, 0)
    
    # Merge Alpha
    b, g, r = cv2.split(sig_color)
    final_png = cv2.merge([b, g, r, alpha_mask])
    return final_png, alpha_mask

def remove_background_threshold(image_pil, threshold_val):
    """ Removes light/dark backgrounds using Grayscale Thresholding """
    img_np = np.array(image_pil)
    # Convert from RGB (PIL) to BGR (OpenCV)
    if img_np.shape[2] == 4: 
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else: img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 

    # 1. Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Create Mask
    _, alpha_mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Merge
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, alpha_mask])

def remove_green_screen(image_pil, sensitivity):
    """ Removes Green pixels using HSV Color Masking """
    img_np = np.array(image_pil)
    # Convert to HSV (Hue, Saturation, Value) to find 'Green'
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # Define Green Range (OpenCV Hue is 0-179, Green is ~60)
    # Sensitivity widens the range to catch light/dark green
    lower_green = np.array([60 - sensitivity, 40, 40])
    upper_green = np.array([60 + sensitivity, 255, 255])
    
    # Create Mask: 255 where Green is found, 0 elsewhere
    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    
    # Invert Mask: We want Green to be TRANSPARENT (0)
    mask_inv = cv2.bitwise_not(mask)
    
    # Convert original to BGR for output
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img_bgr)
    
    # Merge with inverted mask
    return cv2.merge([b, g, r, mask_inv])

# ==========================================
# MAIN APP
# ==========================================

st.set_page_config(page_title="Image Processing Studio", layout="wide")
st.title("🎨 Module 3 Watermarking & Transparancy")
# --- START OF DESCRIPTION ---
st.markdown(
    """
    **Project Demonstration:** This application demonstrates how to create a transparent digital signature 
    from a photo and place it onto a document. 
    
    It also includes an interactive **Background Remover** tool with two modes: 
    
    * **Thresholding** (for white/dark backgrounds)
    * **Green Chroma Screen** (for green backgrounds)
    """
)
st.markdown(
    """This application combines techniques from **Thresholding**, **Logical Operations**, and **Alpha Channels**."""
)
# --- END OF NEW DESCRIPTION ---

tab1, tab2, tab3 = st.tabs(["✒️ E-Signature", "📄 Watermark", "✂️ Background Remover"])

# --- TAB 1: E-SIGNATURE ---
with tab1:
    st.header("1. Create Digital Signature")
    sig_file = st.file_uploader("Upload Signature", type=['jpg', 'png', 'jpeg'], key="sig")
    if sig_file:
        original = Image.open(sig_file)
        c1, c2 = st.columns(2)
        with c1:
            st.image(original, caption="Original", use_container_width=True)
            thresh = st.slider("Threshold", 0, 255, 150, key="s_t")
            color = st.color_picker("Ink Color", "#0000FF", key="s_c")
            intensity = st.slider("Tint", 0.0, 1.0, 1.0, key="s_i")
        result, mask = process_signature(original, thresh, color, intensity)
        with c2:
            st.image(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA), caption="Result", use_container_width=True)
            st.download_button("Download", convert_image_to_bytes(result), "signature.png", "image/png")

# --- TAB 2: WATERMARK ---
with tab2:
    st.header("2. Watermark Document")
    c_main, c_ctrl = st.columns([2, 1])
    with c_ctrl:
        main_doc = st.file_uploader("Document", type=['jpg', 'png'], key="wm_d")
        wm_logo = st.file_uploader("Watermark", type=['png'], key="wm_l")
        if main_doc and wm_logo:
            scale = st.slider("Size", 0.1, 2.0, 0.3)
            opacity = st.slider("Opacity", 0.0, 1.0, 0.8)
            x_pos = st.slider("X %", 0, 100, 80)
            y_pos = st.slider("Y %", 0, 100, 80)
    with c_main:
        if main_doc and wm_logo:
            base = Image.open(main_doc).convert("RGBA")
            watermark = Image.open(wm_logo).convert("RGBA")
            w, h = watermark.size
            watermark = watermark.resize((int(w * scale), int(h * scale)))
            r, g, b, a = watermark.split()
            a = a.point(lambda p: int(p * opacity))
            watermark = Image.merge('RGBA', (r, g, b, a))
            bw, bh = base.size
            x = int((bw - watermark.size[0]) * (x_pos/100))
            y = int((bh - watermark.size[1]) * (y_pos/100))
            final = Image.new('RGBA', base.size)
            final = Image.alpha_composite(base, final)
            base.paste(watermark, (x, y), watermark)
            st.image(base, use_container_width=True)
            buf = io.BytesIO()
            base.save(buf, format="PNG")
            st.download_button("Download Result", buf.getvalue(), "watermarked.png", "image/png")

# --- TAB 3: BACKGROUND REMOVER (UPDATED) ---
with tab3:
    st.header("3. Background Remover")
    
    # Mode Selection
    mode = st.radio("Removal Mode:", ["Standard (Remove White/Dark)", "Green Screen (Chroma Key)"], horizontal=True)
    
    bg_file = st.file_uploader("Upload Image", type=['jpg', 'png'], key="bg")
    
    if bg_file:
        bg_orig = Image.open(bg_file)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(bg_orig, caption="Original", use_container_width=True)
            
            # Show different sliders based on mode
            if "Green Screen" in mode:
                st.info("Adjust sensitivity to catch more shades of green.")
                sensitivity = st.slider("Green Sensitivity", 5, 50, 20)
                result_bg = remove_green_screen(bg_orig, sensitivity)
            else:
                st.info("Adjust threshold to remove light backgrounds.")
                bg_thresh = st.slider("Threshold", 0, 255, 150, key="bg_t")
                result_bg = remove_background_threshold(bg_orig, bg_thresh)
        
        with col_b:
            st.image(cv2.cvtColor(result_bg, cv2.COLOR_BGRA2RGBA), caption="Transparent Result", use_container_width=True)
            st.download_button("Download PNG", convert_image_to_bytes(result_bg), "transparent.png", "image/png")
