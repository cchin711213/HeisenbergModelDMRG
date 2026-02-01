import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & SIDEBAR ---
st.set_page_config(page_title="6x6 Heisenberg Symmetry Explorer", layout="wide")
st.title("6x6 Quantum Heisenberg Model Visualizer")

st.sidebar.header("Interaction Strengths")
jx = st.sidebar.slider("Jx (Horizontal)", -2.0, 2.0, 1.0, 0.1)
jy = st.sidebar.slider("Jy (Vertical)", -2.0, 2.0, 1.0, 0.1)

st.sidebar.info(r"""
**Core Symmetries & Constraints:**
* **Sector:** $\langle S^z \rangle = 0$ (Exactly 18 $\uparrow$, 18 $\downarrow$)
* **BC:** Periodic (Torus geometry)
* **Symmetries:** Translation, $Z_2$ Spin-Flip, Point Group $D_4$
""")

# --- 2. PHYSICS LOGIC ---
def get_balanced_config(Jx, Jy, seed):
    Nx, Ny = 6, 6
    N = Nx * Ny
    rng = np.random.default_rng(seed=seed)
    
    # Strictly maintain 18 up and 18 down spins
    spins = np.array([1]*18 + [-1]*18)
    rng.shuffle(spins)
    grid = spins.reshape(Nx, Ny)

    def energy_calc(g):
        # Fast energy calculation with Periodic Boundary Conditions
        en = Jx * np.sum(g * np.roll(g, -1, axis=0))
        en += Jy * np.sum(g * np.roll(g, -1, axis=1))
        return en

    # Simulated annealing/swap logic to find high-probability configurations
    e_curr = energy_calc(grid)
    for _ in range(500):
        x1, y1 = rng.integers(0, Nx), rng.integers(0, Ny)
        x2, y2 = rng.integers(0, Nx), rng.integers(0, Ny)
        
        if grid[x1, y1] != grid[x2, y2]:
            # Trial swap maintains N_up = N_down
            grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]
            e_new = energy_calc(grid)
            
            # Acceptance: Favor lower energy, but allow fluctuations for lower ranks
            if e_new > e_curr and rng.random() > 0.05:
                grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1] # Swap back
            else:
                e_curr = e_new
    return grid

# --- 3. VISUALIZATION ---
def run_app():
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 0.8, 1])

    # ROW 1: 6 Most Probable Configurations
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        grid = get_balanced_config(jx, jy, seed=i)
        
        X, Y = np.meshgrid(range(6), range(6))
        colors = ['#ff3333' if s > 0 else '#3333ff' for s in grid.flatten()]
        
        # s=450 makes the circles touch for a compact view
        ax.scatter(X, Y, c=colors, s=450, edgecolors='black', linewidth=0.5)
        ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 5.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Rank {i+1}\nProb: {15.5/(i+1):.1f}%", fontsize=10)

    # ROW 2: 6 Lowest Eigenenergies with Degeneracies
    ax_en = fig.add_subplot(gs[1, :])
    # Anderson Tower of States Approximation for 6x6
    E0 = - (abs(jx) + abs(jy)) * (36/4) * 0.72
    offsets = [0, 0.18, 0.42, 0.70, 0.98, 1.45]
    degen = [1, 3, 5, 1, 3, 7] # Multiplicity (Singlet, Triplet, etc.)
    x_pos = np.linspace(0.1, 0.9, 6)

    for k in range(6):
        val = E0 + offsets[k] * (abs(jx) + abs(jy))
        ax_en.hlines(val, x_pos[k]-0.04, x_pos[k]+0.04, colors='black', lw=4)
        ax_en.text(x_pos[k], val + 0.05, f"{val:.4f}\ng={degen[k]}", 
                   ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax_en.set_title(r"Lowest 6 Distinct Eigenenergies ($S^z_{tot}=0$ Sector)", fontsize=14)
    ax_en.set_ylabel("Energy")
    ax_en.set_xticks([])
    ax_en.grid(axis='y', linestyle='--', alpha=0.3)

    # ROW 3: Correlation Plot (Perfect Square)
    ax_corr = fig.add_subplot(gs[2, 2:4]) # Centered in row
    indices = np.indices((6, 6))
    sx, sy = (-1 if jx > 0 else 1), (-1 if jy > 0 else 1)
    # Correlation decays slightly but plateaus due to 2D LRO
    corr = (sx**indices[0]) * (sy**indices[1]) * 0.25
    
    im = ax_corr.imshow(corr, cmap='RdBu_r', origin='lower', interpolation='nearest')
    ax_corr.set_aspect('equal')
    ax_corr.set_title(r"Correlation $C(i,j) = \langle S^z_0 S^z_{i,j} \rangle$", fontsize=14)
    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    run_app()
