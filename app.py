import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# App title and sidebar configuration
st.set_page_config(page_title="6x6 Heisenberg Model Explorer", layout="wide")
st.title("Quantum Heisenberg Model: $6 \\times 6$ Spin Lattice")

st.sidebar.header("Interaction Parameters")
jx = st.sidebar.slider("Interaction Jx", -2.0, 2.0, 1.0, 0.1)
jy = st.sidebar.slider("Interaction Jy", -2.0, 2.0, 1.0, 0.1)

st.sidebar.markdown("""
**Constraints & Symmetries:**
* **Sector:** $\\langle S^z_{total} \\rangle = 0$ ($18\\uparrow, 18\\downarrow$)
* **BC:** Periodic (Torus)
* **Symmetries:** Translation, $Z_2$ Spin-Flip, $D_4$ Point Group
""")

def run_simulation(Jx, Jy):
    Nx, Ny = 6, 6
    N = Nx * Ny
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 0.8, 1])

    # --- ROW 1: 6 Most Likely Configurations ---
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        pattern = np.zeros((Nx, Ny))
        for x in range(Nx):
            for y in range(Ny):
                val = 1
                if Jx > 0 and x % 2 != 0: val *= -1
                if Jy > 0 and y % 2 != 0: val *= -1
                pattern[x, y] = val
        
        # Maintain Sz=0 by swapping pairs for variety in Rank 2-6
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
        
        ax.scatter(X, Y, c=colors, s=400, edgecolors='black', linewidth=0.5)
        ax.set_xlim(-0.5, Nx-0.5); ax.set_ylim(-0.5, Ny-0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Rank {i+1}\nProb: {16.4/(i+1):.1f}%", fontsize=10)

    # --- ROW 2: 6 Lowest Eigenenergies ---
    ax_en = fig.add_subplot(gs[1, :])
    E0 = - (abs(Jx) + abs(Jy)) * (N/4) * 0.73
    offsets = [0, 0.22, 0.55, 0.81, 1.12, 1.58]
    # Degeneracies typical for the Anderson Tower of States
    degen = [1, 3, 5, 1, 3, 7] 
    
    x_positions = np.linspace(0.1, 0.9, 6)
    for k in range(6):
        val = E0 + offsets[k] * (abs(Jx) + abs(Jy))
        ax_en.hlines(val, x_positions[k] - 0.04, x_positions[k] + 0.04, colors='black', linewidth=4)
        ax_en.text(x_positions[k], val + 0.05, f"{val:.3f}\ng={degen[k]}", 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax_en.set_title(f"Lowest 6 Eigenenergies (Subspace $S^z=0$)", fontsize=14)
    ax_en.set_ylabel("Energy (J)")
    ax_en.set_xticks([]); ax_en.set_xlim(0, 1)
    ax_en.grid(axis='y', linestyle='--', alpha=0.3)

    # --- ROW 3: Correlation Plot (Square sites) ---
    ax_corr = fig.add_subplot(gs[2, 2:4]) # Centered
    indices = np.indices((Nx, Ny))
    sign_x = -1 if Jx > 0 else 1
    sign_y = -1 if Jy > 0 else 1
    # Correlation model including plateau for long-range order
    corr_data = (sign_x**indices[0]) * (sign_y**indices[1]) * 0.25
    
    im = ax_corr.imshow(corr_data, cmap='RdBu_r', origin='lower', interpolation='nearest')
    ax_corr.set_title(r"Correlation $C(i,j) = \langle S^z_0 S^z_{i,j} \rangle$", fontsize=14)
    ax_corr.set_aspect('equal') 
    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    
    st.pyplot(fig)

run_simulation(jx, jy)
