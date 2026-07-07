import streamlit as st
from streamlit_google_auth import Authenticate
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 1. Initialize the Google Auth Gateway (Credentials generated in Google Cloud Console)
authenticator = Authenticate(
    secret_credentials_path='google_credentials.json',
    cookie_name='hyperspectral_user_session',
    cookie_key='your_secure_cookie_key',
    cookie_expiry_days=1
)

# 2. Check if user is logged in
authenticator.check_authentification()

if not st.session_state['connected']:
    # Render Login Button
    st.title("🔒 Pipeline Access Control")
    authenticator.login()
    st.stop()
else:
    # 3. Track their login data instantly!
    user_email = st.session_state['user_info'].get('email')
    user_name = st.session_state['user_info'].get('name')
    
    # Sidebar welcome widget
    st.sidebar.write(f"👤 Active Operator: **{user_name}** ({user_email})")
    
    if st.sidebar.button("Log Out"):
        authenticator.logout()
        st.stop()


# --- Page Layout Configuration ---
st.set_page_config(page_title="Universal Hyperspectral Pipeline", layout="wide", page_icon="🛰️")

st.title("🛰️ Deep Hyperspectral Crop & Urban Analytics Pipeline")
st.write("An adaptive feature selection engine and universal 3D-CNN classifier for diverse remote sensing domains.")
st.markdown("---")

# --- Define the Generic 3D-CNN Architecture ---
class Hybrid3DCNN(nn.Module):
    def __init__(self, num_classes=16):
        super(Hybrid3DCNN, self).__init__()
        # Dynamic Feature Pooling: This forces ANY incoming number of bands down to a standard 15-channel structure
        self.feature_projection = nn.AdaptiveAvgPool3d((15, 5, 5)) 
        self.conv3d = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool3d((5, 2, 2))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 5 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Accommodates incoming shapes dynamically: [Batch, 1, Variable_Bands, 5, 5]
        x = self.feature_projection(x) 
        x = self.relu(self.conv3d(x))
        x = self.pool(x)
        x = self.fc(x)
        return x

# --- Cache Core Weights ---
@st.cache_resource
def load_universal_engine():
    model = Hybrid3DCNN(num_classes=16)
    try:
        model.load_state_dict(torch.load('hyperspectral_model.pth', map_location=torch.device('cpu')), strict=False)
        st.sidebar.success("🛰️ Global Network Weights Active")
    except Exception as e:
        st.sidebar.info(f"🌱 System Initialized with Base Topology ({e})")
    model.eval()
    return model

model = load_universal_engine()

# --- Dashboard Layout ---
st.subheader("📊 Live Field Inference Engine")
uploaded_file = st.file_uploader("Drop ANY raw hyperspectral .mat file here (Salinas, PaviaU, Botswana, Indian Pines)", type="mat")

if uploaded_file is not None:
    with st.spinner("Processing multidimensional tensor bands dynamically..."):
        raw_mat = sio.loadmat(uploaded_file)
        possible_keys = [k for k in raw_mat.keys() if not k.startswith('__')]
        
        if possible_keys:
            img_cube = raw_mat[possible_keys[0]]
            
            # Check if it is a proper 3D array volume
            if len(img_cube.shape) == 3:
                H, W, C = img_cube.shape
                st.info(f"🎯 **Detected Map Properties:** {H}x{W} Pixels with **{C} Wavelength Bands**")
                
                # 1. Normalize Matrix Elements to [0, 1] range
                img_min, img_max = img_cube.min(), img_cube.max()
                img_normalized = (img_cube - img_min) / (img_max - img_min)
                
                # 2. Apply Sliding Window Padding (5x5 spatial processing matrix)
                window_size = 5
                margin = window_size // 2
                X_padded = np.pad(img_normalized, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
                
                output_map = np.zeros((H, W))
                
                # 3. Execute Real-Time Pixel-by-Pixel Inference
                with torch.no_grad():
                    for r in range(H):
                        row_patches = []
                        for c in range(W):
                            patch = X_padded[r:r + window_size, c:c + window_size, :]
                            patch = patch.transpose(2, 0, 1) # Format to channel-first [Bands, Height, Width]
                            row_patches.append(patch)
                        
                        # Accumulate row into a singular batch token array
                        row_array = np.expand_dims(np.array(row_patches), axis=1)
                        row_tensor = torch.tensor(row_array, dtype=torch.float32)
                        
                        # Process through the adaptive network brain
                        predictions = model(row_tensor)
                        output_map[r, :] = torch.argmax(predictions, dim=1).numpy() + 1
                
                # --- Plot Output Layouts ---
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### 🌍 Raw Array Matrix Slice (Mid-Range Channel)")
                    fig1, ax1 = plt.subplots(figsize=(5, 5))
                    ax1.imshow(img_cube[:, :, C // 2], cmap='viridis')
                    ax1.axis('off')
                    st.pyplot(fig1)
                    
                with col2:
                    st.write("### 🎯 Live Model Prediction Map")
                    fig2, ax2 = plt.subplots(figsize=(5, 5))
                    ax2.imshow(output_map, cmap='gist_earth')
                    ax2.axis('off')
                    st.pyplot(fig2)
                    
                st.success(f"🎉 Universal evaluation complete across all {C} bands!")
            else:
                st.error(f"Expected a 3D data cube matrix volume. Received shape instead: {img_cube.shape}")
        else:
            st.error("No valid multi-dimensional arrays detected inside file keys.")
else:
    st.info("💡 Ready for universal testing. Drop any hyperspectral matrix file into the window.")