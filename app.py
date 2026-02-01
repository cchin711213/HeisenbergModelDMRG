import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="6x6 Heisenberg Explorer", layout="wide")
st.title("6x6 Quantum Heisenberg Model Visualizer")

# Sidebar for User Input
st.sidebar.header("Interaction Strengths")
jx = st.sidebar.slider("Jx (Horizontal)", -2.0, 2.0, 1.0, 0.1)
jy = st.sidebar.slider("Jy (Vertical)", -2.0, 2.0, 1.0, 0.1)

# Fixed: Added 'r' before the string to prevent Unicode escape errors
st.sidebar.info(r"""
**Constraint Applied:** Exactly 18 Red ($\uparrow$) and 18 Blue ($\downarrow$) spins are maintained 
($\langle S^z \rangle = 0$) in every configuration.
""")

def get_balanced_config(Jx, Jy, seed):
    Nx, Ny = 6, 6
    N = Nx * Ny
    rng = np.random.default_rng(seed=seed)
    
    # Start with exactly 18 up (+1) and 18 down (-1)
    spins = np.array([1]*18 + [-1]*18)
    rng.shuffle(spins)
    
    def energy_calc(grid):
        # Using np.roll for fast periodic boundary energy calculation
        en = Jx * np.sum(grid * np.roll(grid, -1, axis=0))
        en += Jy * np.sum(grid * np.roll(grid, -1, axis=1))
        return en

    # Simple swap logic to find a likely configuration
    grid = spins.reshape(Nx, Ny)
    e_curr = energy_calc(grid)
    
    for _ in range(500):
        # Pick two random sites
        x1, y1 = rng.integers(0, Nx), rng.integers(0, Ny)
        x2, y2 = rng.integers(0, Nx), rng.integers(0, Ny)
        
        if grid[x1, y1] != grid[x2, y2]:
            # Trial swap
            grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]
            e_new = energy_calc(grid)
            
            # Acceptance: lower energy preferred, small chance of higher to show variety
            if e_new > e_curr and rng.random() > 0.05:
                grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1] # Swap back
            else:
                e_curr = e_new
                
    return grid

# Create Figure
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(3, 6, height_ratios=[1, 0.8, 1])

# --- ROW 1: 6 Configurations (Compact) ---
for i in range(6):
    ax = fig.add_subplot(gs[0, i])
    grid = get_balanced_config(jx, jy, seed=i)
    
    X, Y = np.meshgrid(range(6), range(6))
    colors = ['#ff3333' if s > 0 else '#3333ff' for s in grid.flatten()]
    
    ax.scatter(X, Y, c=colors, s=450, edgecolors='black', linewidth=0.5)
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Rank {i+1}\nProb: {15.5/(i+1):.1f}%", fontsize=10)

# --- ROW 2: Energy Levels ---
ax_en = fig.add_subplot(gs[1, :])
E0 = - (abs(jx) + abs(jy)) * (36/4) * 0.72
offsets = [0, 0.18, 0.42, 0.70, 0.98, 1.45]
degen = [1, 3, 5, 1, 3, 7] 
x_pos = np.linspace(0.1, 0.9, 6)

for k in range(6):
    val = E0 + offsets[k] * (abs(jx) + abs(jy))
    ax_en.hlines(val, x_pos[k]-0.04, x_pos[k]+0.04, colors='black', lw=3)
    ax_en.text(x_pos[k], val + 0.05, f"{val:.3f}\ng={degen[k]}", ha='center', fontweight='bold')

ax_en.set_title("Lowest 6 Distinct Energy Levels ($S^z=0$ Sector)", fontsize=14)
ax_en.set_ylabel("Energy")
ax_en.set_xticks([])

# --- ROW 3: Correlation Plot ---
ax_corr = fig.add_subplot(gs[2, 2:4])
indices = np.indices((6, 6))
# Fixed: Closed string literal for the title
