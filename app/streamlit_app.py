"""
Streamlit interface for fruit calibration.
Allows switching between segmentation engines, adjusting caliber dimensions,
uploading an image, and displaying results.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.segmentation.factory import SegmentationFactory
from src.dimension import DimensionCalculator
from src.display import Visualizer

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Fruit Calibration",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF6B35;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF6B35;
        margin-bottom: 1rem;
    }
    .result-success {
        border-left-color: #28a745;
    }
    .result-fail {
        border-left-color: #dc3545;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #333;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .status-pass {
        color: #28a745;
        font-weight: 700;
    }
    .status-fail {
        color: #dc3545;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Initialize session state
# ----------------------------------------------------------------------
if "engine" not in st.session_state:
    st.session_state.engine = None
if "dim_calc" not in st.session_state:
    st.session_state.dim_calc = DimensionCalculator()

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3076/3076105.png",
        width=80,
    )
    st.markdown("## ⚙️ Controls")

    # Segmentation mode
    mode = st.radio(
        "Segmentation engine",
        options=["HSV", "UNet"],
        index=0,
        help="HSV uses color thresholds; UNet uses a trained deep learning model.",
    )

    # UNet checkpoint path (only shown if UNet selected)
    unet_checkpoint = None
    if mode == "UNet":
        unet_checkpoint = st.text_input(
            "UNet checkpoint path",
            value="src/models/Unet-98.ckpt",
            help="Path to the trained UNet model checkpoint.",
        )

    st.markdown("---")

    # Caliber dimensions (in mm)
    st.markdown("### 📏 Caliber standards")
    min_length = st.slider(
        "Min length (mm)",
        min_value=50,
        max_value=150,
        value=80,
        step=1,
    )
    max_length = st.slider(
        "Max length (mm)",
        min_value=50,
        max_value=200,
        value=120,
        step=1,
    )
    min_width = st.slider(
        "Min width (mm)",
        min_value=30,
        max_value=100,
        value=60,
        step=1,
    )
    max_width = st.slider(
        "Max width (mm)",
        min_value=30,
        max_value=120,
        value=80,
        step=1,
    )

    st.markdown("---")
    st.caption("Upload an image to start analysis.")

# ----------------------------------------------------------------------
# Create engine based on mode
# ----------------------------------------------------------------------
def get_engine():
    """Create or retrieve the segmentation engine based on current mode."""
    if mode == "HSV":
        return SegmentationFactory.create(engine_type="hsv")
    else:  # UNet
        if unet_checkpoint:
            return SegmentationFactory.create(
                engine_type="unet",
                checkpoint_path=unet_checkpoint,
                device="cpu",  # or "cuda" if available
                smooth_kernel=15,
            )
        else:
            st.error("Please provide a valid checkpoint path for UNet.")
            return None

# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
st.markdown('<div class="main-header">🥭 Fruit Calibration</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload an image, adjust settings, and get instant dimensions.</div>',
    unsafe_allow_html=True,
)

# Image upload
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

# Placeholder for image and results
col1, col2 = st.columns([2, 1], gap="medium")

# Left column: original image + processed overlay
with col1:
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Get the segmentation engine
        engine = get_engine()
        dim_calc = st.session_state.dim_calc
        vis = Visualizer(contour_thickness=15, text_scale=2.0, text_thickness=5)

        if engine is not None:
            try:
                # Process the image
                contours = engine.get_contours(image)
                dimensions_cm = dim_calc.compute_dimensions(contours)

                # Generate visualization
                display = vis.display_results(image, contours, dimensions_cm)
                display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

                # Display the result
                st.image(display_rgb, caption="Processed result", use_container_width=True)

                # Extract mango dimensions for display
                mango_pts, (mango_long_cm, mango_larg_cm) = dimensions_cm.get("mango", (None, (0, 0)))
                card_pts, (card_long_cm, card_larg_cm) = dimensions_cm.get("card", (None, (0, 0)))

                # Determine if within standards
                if mango_long_cm > 0 and mango_larg_cm > 0:
                    length_ok = min_length <= mango_long_cm * 10 <= max_length  # convert cm to mm
                    width_ok = min_width <= mango_larg_cm * 10 <= max_width
                    status = "✅ PASS" if (length_ok and width_ok) else "❌ FAIL"
                    status_class = "status-pass" if (length_ok and width_ok) else "status-fail"
                else:
                    status = "⚠️ Not detected"
                    status_class = "status-fail"
                    length_ok = False
                    width_ok = False

            except Exception as e:
                st.error(f"Error during processing: {e}")
                status = "⚠️ Error"
                status_class = "status-fail"
                mango_long_cm = mango_larg_cm = 0
                length_ok = width_ok = False
        else:
            status = "⚠️ Engine not available"
            status_class = "status-fail"
            mango_long_cm = mango_larg_cm = 0
            length_ok = width_ok = False

        # ------------------------------------------------------------------
        # Right column: Results
        # ------------------------------------------------------------------
        with col2:
            st.markdown("### 📊 Results")

            # Display metrics
            st.markdown(
                f"""
                <div class="result-box {'result-success' if 'PASS' in status else 'result-fail' if 'FAIL' in status else ''}">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span class="metric-label">Engine</span>
                        <span class="metric-value">{mode}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span class="metric-label">Caliber (L × W)</span>
                        <span class="metric-value">{min_length}–{max_length} mm × {min_width}–{max_width} mm</span>
                    </div>
                    <hr>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                        <span class="metric-label">Measured L × W</span>
                        <span class="metric-value">{mango_long_cm*10:.1f} × {mango_larg_cm*10:.1f} mm</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                        <span class="metric-label">Length</span>
                        <span style="color: {'#28a745' if length_ok else '#dc3545'};">
                            {'✅' if length_ok else '❌'} {mango_long_cm*10:.1f} mm
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                        <span class="metric-label">Width</span>
                        <span style="color: {'#28a745' if width_ok else '#dc3545'};">
                            {'✅' if width_ok else '❌'} {mango_larg_cm*10:.1f} mm
                        </span>
                    </div>
                    <hr>
                    <div style="display: flex; justify-content: space-between; margin-top: 0.3rem;">
                        <span class="metric-label" style="font-size: 1.1rem;">Status</span>
                        <span class="{status_class}" style="font-size: 1.2rem;">{status}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Additional info about card detection
            if card_long_cm > 0 and card_larg_cm > 0:
                st.caption(f"Reference card: {card_long_cm:.1f} × {card_larg_cm:.1f} cm")
            else:
                st.caption("⚠️ Reference card not detected")

    else:
        st.info("👈 Upload an image to begin analysis.")
        with col2:
            st.markdown("### 📊 Results")
            st.markdown(
                """
                <div class="result-box">
                    <em>No image loaded yet.</em>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit · Fruit Calibration System v1.0")
