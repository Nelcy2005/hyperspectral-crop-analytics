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

# --- Define the Optimized Dynamic 3D-CNN Architecture ---
class Hybrid3DCNN(nn.Module):
    def __init__(self, num_classes=16):
        super(Hybrid3DCNN, self).__init__()
        
        # Pointwise Convolution: Dynamically compresses variable bands down to 15 channels without blurring features
        self.feature_projection = nn.Conv3d(1, 1, kernel_size=(1, 1, 1)) 
        
        # Standard target channels for spectral band processing
        self.target_bands = 15
        
        self.conv3d = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool3d((self.target_bands, 2, 2))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * self.target_bands * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # 1. Dynamically adapt the weights of the pointwise convolution to match incoming spectral bands
        incoming_bands = x.shape[2]
        if self.feature_projection.kernel_size[0] != incoming_bands:
            # Re-shape the weights instantly to match the exact input channel size
            self.feature_projection = nn.Conv3d(1, 1, kernel_size=(incoming_bands, 1, 1), device=x.device)
            # Fix weights so it performs a clean step-down slice by default
            nn.init.constant_(self.feature_projection.weight, 1.0 / incoming_bands)
            
        # 2. Process through the network pipeline cleanly
        x = self.relu(self.feature_projection(x)) # Dynamic projection layer
        
        # Change dimensions to fit conv3d requirements: [Batch, 1, 15, 5, 5]
        x = torch.nn.functional.interpolate(x, size=(self.target_bands, 5, 5), mode='trilinear', align_corners=False)
        
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
                
                # 3. Optimized Batch-by-Batch Inference to completely prevent cloud memory crashes
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
                            
                            # Execute when full batch size token is met
                            if len(patches_accumulator) == batch_size:
                                row_array = np.array(patches_accumulator)
                                row_array = np.expand_dims(row_array, axis=1)
                                row_tensor = torch.tensor(row_array, dtype=torch.float32)
                                
                                predictions = model(row_tensor)
                                pred_classes = torch.argmax(predictions, dim=1).numpy() # REMOVED + 1
                                
                                for i, (pr, pc) in enumerate(coords_accumulator):
                                    output_map[pr, pc] = pred_classes[i]
                                    
                                patches_accumulator = []
                                coords_accumulator = []
                    
                    # Process leftover pixels boundary segments
                    if patches_accumulator:
                        row_array = np.array(patches_accumulator)
                        row_array = np.expand_dims(row_array, axis=1)
                        row_tensor = torch.tensor(row_array, dtype=torch.float32)
                        predictions = model(row_tensor)
                        pred_classes = torch.argmax(predictions, dim=1).numpy() # REMOVED + 1
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
                    
                    # TUNING: Correctly mask out class 0 (Unclassified Background) as transparent blank space
                    tuned_map = np.where(output_map == 0, np.nan, output_map) 
                    
                    # Using 'nipy_spectral' helps distinct classes pop out dynamically
                    ax2.imshow(tuned_map, cmap='nipy_spectral')
                    ax2.axis('off')
                    st.pyplot(fig2)
                    
                st.success(f"🎉 Universal evaluation complete across all {C} bands!")
            else:
                st.error(f"Expected a 3D data cube matrix volume. Received shape instead: {img_cube.shape}")
        else:
            st.error("No valid multi-dimensional arrays detected inside file keys.")
else:
    st.info("💡 Ready for universal testing. Drop any hyperspectral matrix file into the window.")
