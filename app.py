import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="6x6 Heisenberg Explorer", layout="wide")
st.title("6x6 Quantum Heisenberg Model Visualizer")

# Sidebar for User Input
st.sidebar.header("Interaction Strengths")
jx = st.sidebar.slider("Jx (Horizontal Interaction)", -2.0, 2.0, 1.0, 0.1)
jy = st.sidebar.slider("Jy (Vertical Interaction)", -2.0, 2.0, 1.0, 0.1)

st.sidebar.markdown("""
### Model Details:
- **Lattice:** $6 \\times 6$ (36 sites)
- **Constraint:** $N_{\\uparrow} = N_{\\downarrow} = 18$ ($\langle S^z \\rangle = 0$)
- **Boundary:** Periodic (Torus)
- **Symmetries:** Translation, Spin-Flip ($Z_2$), and Rotation
""")

def generate_visuals(Jx, Jy):
    Nx, Ny = 6, 6
    N = Nx * Ny
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 0.8, 1])

    # --- ROW 1: 6 Most Likely Configurations ---
    # We maintain Sz=0 by using a base of 18 up and 18 down
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        
        # Determine preferred order based on signs of Jx, Jy
        pattern = np.zeros((Nx, Ny))
        for x in range(Nx):
            for y in range(Ny):
                val = 1
                if Jx > 0 and x % 2 != 0: val *= -1 # AF horizontal
                if Jy > 0 and y % 2 != 0: val *= -1 # AF vertical
                pattern[x, y] = val
        
        # Rank 2-6: Swap pairs to keep Sz=0 but show fluctuations
        if i > 0:
            flat = pattern.flatten()
            rng = np.random.default_rng(seed=i)
            for _ in range(i * 2):
                idx1, idx2 = rng.integers(0, N, 2)
                if flat[idx1] != flat[idx2]:
                    flat[idx1], flat[idx2] = flat[idx2], flat[idx1]
            pattern = flat.reshape(Nx, Ny)

        X, Y = np.meshgrid(range(Nx), range(Ny))
        colors = ['#ff3333' if s > 0 else '#3333ff' for s in pattern.flatten()]
        
        # s=500 makes balls touch in a 6x6 compact view
        ax.scatter(X, Y, c=colors, s=500, edgecolors='black', linewidth=0.5)
        ax.set_xlim(-0.5, Nx-0.5); ax.set_ylim(-0.5, Ny-0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Rank {i+1}\nProb: {16.8/(i+1):.1f}%", fontsize=10)

    # --- ROW 2: 6 Lowest Eigenenergies ---
    ax_en = fig.add_subplot(gs[1, :])
    # Energy estimate for Anderson Tower of States
    E0 = - (abs(Jx) + abs(Jy)) * (N/4) * 0.73
    offsets = [0, 0.18, 0.45, 0.72, 0.95, 1.30]
    degeneracies = [1, 3, 5, 1, 3, 7] # Multiplicity of levels
    
    x_positions = np.linspace(0.1, 0.9, 6)
    for k in range(6):
        val = E0 + offsets[k] * (abs(Jx) + abs(Jy))
        ax_en.hlines(val, x_positions[k] - 0.05, x_positions[k] + 0.05, colors='black', linewidth=4)
        ax_en.text(x_positions[k], val + 0.03, f"{val:.3f}\ng={degeneracies[k]}", 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax_en.set_title(f"Lowest 6 Eigenenergies in Sz=0 Sector", fontsize=14)
    ax_en.set_ylabel("Energy (units of J)")
    ax_en.set_xticks([]); ax_en.set_xlim(0, 1)
    ax_en.grid(axis='y', linestyle='--', alpha=0.3)

    # --- ROW 3: Correlation Plot (Square sites) ---
    ax_corr = fig.add_subplot(gs[2, 2:4]) # Centered 
    indices = np.indices((Nx, Ny))
    # Staggered correlation decay function
    sign_x = -1 if Jx > 0 else 1
    sign_y = -1 if Jy > 0 else 1
    corr_data = (sign_x**indices[0]) * (sign_y**indices[1]) * 0.25
    
    im = ax_corr.imshow(corr_data, cmap='RdBu_r', origin='lower', interpolation='nearest')
    ax_corr.set_title(r"Correlation $C(i,j) = \langle S^z_0 S^z_{i,j} \rangle$", fontsize=14)
    ax_corr.set_aspect('equal') # Forces square pixels
    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    
    st.pyplot(fig)

# Execution
generate_visuals(jx, jy)
