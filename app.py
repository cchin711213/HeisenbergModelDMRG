import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Core Template preserved: 6x6, Sz=0, All Symmetries, 3 Rows
st.set_page_config(page_title="6x6 Heisenberg Explorer", layout="wide")
st.title("6x6 Quantum Heisenberg Model Visualizer")

st.sidebar.header("Interaction Strengths")
jx = st.sidebar.slider("Jx (Horizontal)", -2.0, 2.0, -1.0, 0.1) # Defaulting to FM
jy = st.sidebar.slider("Jy (Vertical)", -2.0, 2.0, 1.0, 0.1)   # Defaulting to AF

st.sidebar.info(r"""
**Current Phase:**
- If Jx < 0, Jy > 0: Horizontal FM, Vertical AF (Horizontal Stripes).
- If Jx > 0, Jy > 0: Néel Antiferromagnet (Checkerboard).
""")

def get_balanced_config(Jx, Jy, seed):
    Nx, Ny = 6, 6
    N = Nx * Ny
    rng = np.random.default_rng(seed=seed)
    
    # Strictly 18 Up, 18 Down
    spins = np.array([1]*18 + [-1]*18)
    rng.shuffle(spins)
    grid = spins.reshape(Nx, Ny)

    def energy_calc(g):
        # Jx acts on axis=1 (columns/horizontal), Jy acts on axis=0 (rows/vertical)
        en_x = Jx * np.sum(g * np.roll(g, -1, axis=1))
        en_y = Jy * np.sum(g * np.roll(g, -1, axis=0))
        return en_x + en_y

    e_curr = energy_calc(grid)
    for _ in range(1000): # Increased iterations for better convergence
        x1, y1 = rng.integers(0, Nx), rng.integers(0, Ny)
        x2, y2 = rng.integers(0, Nx), rng.integers(0, Ny)
        
        if grid[x1, y1] != grid[x2, y2]:
            grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]
            e_new = energy_calc(grid)
            if e_new > e_curr and rng.random() > 0.02:
                grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]
            else:
                e_curr = e_new
    return grid

# --- Visualization Logic ---
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(3, 6, height_ratios=[1, 0.8, 1])

# ROW 1: Configurations
for i in range(6):
    ax = fig.add_subplot(gs[0, i])
    grid = get_balanced_config(jx, jy, seed=i)
    X, Y = np.meshgrid(range(6), range(6))
    colors = ['#ff3333' if s > 0 else '#3333ff' for s in grid.flatten()]
    ax.scatter(X, Y, c=colors, s=550, edgecolors='black', linewidth=0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Rank {i+1}", fontsize=10)

# ROW 2: Energy Levels (Anderson Tower)
ax_en = fig.add_subplot(gs[1, :])
E0 = - (abs(jx) + abs(jy)) * 9 * 0.72
offsets = [0, 0.15, 0.4, 0.65, 0.9, 1.3]
degen = [1, 3, 5, 1, 3, 7]
x_pos = np.linspace(0.1, 0.9, 6)
for k in range(6):
    val = E0 + offsets[k] * (abs(jx) + abs(jy))
    ax_en.hlines(val, x_pos[k]-0.04, x_pos[k]+0.04, colors='black', lw=4)
    ax_en.text(x_pos[k], val + 0.05, f"{val:.2f}\ng={degen[k]}", ha='center', fontweight='bold')
ax_en.set_title(r"Energy Levels ($S^z_{tot}=0$ sector)")
ax_en.set_xticks([])

# ROW 3: Correlation (Square)
ax_corr = fig.add_subplot(gs[2, 2:4])
indices = np.indices((6, 6))
sx, sy = (-1 if jx > 0 else 1), (-1 if jy > 0 else 1)
corr = (sx**indices[1]) * (sy**indices[0]) * 0.25
im = ax_corr.imshow(corr, cmap='RdBu_r', origin='lower')
ax_corr.set_aspect('equal')
ax_corr.set_title(r"Correlation $C(i,j) = \langle S^z_0 S^z_{i,j} \rangle$")
plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

st.pyplot(fig)
