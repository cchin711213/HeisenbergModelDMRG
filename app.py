import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="DMRG Heisenberg 6x6", layout="wide")

# Safe Title Implementation
st.title("DMRG Simulation of Heisenberg Model")
st.latex(r"H = J_x \sum_{\langle i,j \rangle_x} \mathbf{S}_i \cdot \mathbf{S}_j + J_y \sum_{\langle i,j \rangle_y} \mathbf{S}_i \cdot \mathbf{S}_j")

# Sidebar Controls
st.sidebar.header("Interaction Parameters")
jx = st.sidebar.slider("Jx (Horizontal)", -2.0, 2.0, 1.0, 0.1)
jy = st.sidebar.slider("Jy (Vertical)", -2.0, 2.0, 1.0, 0.1)

st.sidebar.info(r"""
**System Specifications:**
- **Lattice:** $6 \times 6$ Torus (PBC)
- **Constraint:** $N_\uparrow = N_\downarrow = 18$
- **Symmetry:** $S^z_{total} = 0$
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
        # Axis 1 is horizontal (Jx), Axis 0 is vertical (Jy)
        en_x = Jx * np.sum(g * np.roll(g, -1, axis=1))
        en_y = Jy * np.sum(g * np.roll(g, -1, axis=0))
        return en_x + en_y

    e_curr = energy_calc(grid)
    for _ in range(800):
        x1, y1 = rng.integers(0, Nx), rng.integers(0, Ny)
        x2, y2 = rng.integers(0, Nx), rng.integers(0, Ny)
        
        if grid[x1, y1] != grid[x2, y2]:
            grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]
            e_new = energy_calc(grid)
            if e_new > e_curr and rng.random() > 0.03:
                grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]
            else:
                e_curr = e_new
    return grid

# --- 3. VISUALIZATION ---
def run_app():
    # Force a fresh figure to prevent overlapping plots
    plt.clf()
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 0.8, 1])

    # ROW 1: 6 Configurations (Full balls visible)
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        grid = get_balanced_config(jx, jy, seed=i)
        
        X, Y = np.meshgrid(range(6), range(6))
        colors = ['#ff3333' if s > 0 else '#3333ff' for s in grid.flatten()]
        
        # s=550 with the buffer ensures balls are full and touching
        ax.scatter(X, Y, c=colors, s=550, edgecolors='black', linewidth=0.5)
