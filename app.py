import streamlit as st
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- Page Layout Configuration ---
st.set_page_config(page_title="Universal Hyperspectral Pipeline", layout="wide", page_icon="🛰️")

st.title("🛰️ Deep Hyperspectral Crop & Urban Analytics Pipeline")
st.write("An adaptive feature selection engine and universal 3D-CNN classifier for diverse remote sensing domains.")
st.markdown("---")

# --- Define the High-Precision Deep 3D-CNN Architecture (Matches Training Engine) ---
class HighPrecision3DCNN(nn.Module):
    def __init__(self, num_classes=16):
        super(HighPrecision3DCNN, self).__init__()
        
        # Pointwise Convolution
        self.feature_projection = nn.Conv3d(1, 1, kernel_size=(1, 1, 1)) 
        self.target_bands = 15
        
        # Deeper Feature Extractors matching the training engine
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool3d((self.target_bands, 2, 2))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * self.target_bands * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        incoming_bands = x.shape[2]
        if self.feature_projection.kernel_size[0] != incoming_bands:
            self.feature_projection = nn.Conv3d(1, 1, kernel_size=(incoming_bands, 1, 1), device=x.device)
            nn.init.constant_(self.feature_projection.weight, 1.0 / incoming_bands)
            
        x = self.relu(self.feature_projection(x))
        x = torch.nn.functional.interpolate(x, size=(self.target_bands, 5, 5), mode='trilinear', align_corners=False)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.fc(x)
        return x
        
# --- Cache Core Weights ---
@st.cache_resource
def load_universal_engine():
    model = HighPrecision3DCNN(num_classes=16)
    try:
        model.load_state_dict(torch.load('hyperspectral_model.pth', map_location=torch.device('cpu')), strict=True)
        st.sidebar.success("🛰️ Global Network Weights Active")
    except Exception as e:
        st.sidebar.error(f"🌱 Error loading state: {e}")
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
                
                # 3. Batch Inference Pipeline
                batch_size = 512  
                patches_accumulator = []
                coords_accumulator = []
                
                with torch.no_grad():
                    for r in range(H):
                        for c in range(W):
                            patch = X_padded[r:r + window_size, c:c + window_size, :]
                            patch = patch.transpose(2, 0, 1) # Format to channel-first [Bands, Height, Width]
                            patches_accumulator.append(patch)
                            coords_accumulator.append((r, c))
                            
                            if len(patches_accumulator) == batch_size:
                                row_array = np.array(patches_accumulator)
                                row_array = np.expand_dims(row_array, axis=1)
                                row_tensor = torch.tensor(row_array, dtype=torch.float32)
                                
                                predictions = model(row_tensor)
                                pred_classes = torch.argmax(predictions, dim=1).numpy()
                                
                                for i, (pr, pc) in enumerate(coords_accumulator):
                                    output_map[pr, pc] = pred_classes[i]
                                    
                                patches_accumulator = []
                                coords_accumulator = []
                    
                    if patches_accumulator:
                        row_array = np.array(patches_accumulator)
                        row_array = np.expand_dims(row_array, axis=1)
                        row_tensor = torch.tensor(row_array, dtype=torch.float32)
                        predictions = model(row_tensor)
                        pred_classes = torch.argmax(predictions, dim=1).numpy()
                        for i, (pr, pc) in enumerate(coords_accumulator):
                            output_map[pr, pc] = pred_classes[i]
                
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
                    
                    # Display segmented predictions cleanly using nipy_spectral
                    ax2.imshow(output_map, cmap='nipy_spectral')
                    ax2.axis('off')
                    st.pyplot(fig2)
                    
                st.success(f"🎉 Universal evaluation complete across all {C} bands!")
            else:
                st.error(f"Expected a 3D data cube matrix volume. Received shape instead: {img_cube.shape}")
        else:
            st.error("No valid multi-dimensional arrays detected inside file keys.")
else:
    st.info("💡 Ready for universal testing. Drop any hyperspectral matrix file into the window.")
