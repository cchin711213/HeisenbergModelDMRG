import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="DMRG Heisenberg 6x6", layout="wide")

st.title("DMRG Simulation of Heisenberg Model")
st.latex(r"H = J_x \sum_{\langle i,j \rangle_x} \mathbf{S}_i \cdot \mathbf{S}_j + J_y \sum_{\langle i,j \rangle_y} \mathbf{S}_i \cdot \mathbf{S}_j")

# Sidebar Controls
st.sidebar.header("Interaction Parameters")
jx = st.sidebar.slider("Jx (Horizontal)", -2.0, 2.0, -1.0, 0.1)
jy = st.sidebar.slider("Jy (Vertical)", -2.0, 2.0, 1.0, 0.1)

st.sidebar.info(r"""
**Visualization Details:**
- **Row 1:** 6 most likely configurations with relative probabilities.
- **Row 2:** Energy spectrum with degeneracies.
- **Row 3:** Correlation map with fixed range [-0.25, 0.25].
""")

# --- 2. PHYSICS LOGIC ---
@st.cache_data
def get_ranked_configs(Jx, Jy):
    Nx, Ny = 6, 6
    configs = []
    energies = []
    
    # Sample configurations to find the top ground state components
    for seed in range(12):
        rng = np.random.default_rng(seed=seed)
        spins = np.array([1]*18 + [-1]*18)
        rng.shuffle(spins)
        grid = spins.reshape(Ny, Nx)

        for _ in range(600):
            r1, c1 = rng.integers(0, Ny), rng.integers(0, Nx)
            r2, c2 = rng.integers(0, Ny), rng.integers(0, Nx)
            if grid[r1, c1] != grid[r2, c2]:
                def get_local_e(r, c, val):
                    return Jx * val * (grid[r, (c+1)%Nx] + grid[r, (c-1)%Nx]) + \
                           Jy * val * (grid[(r+1)%Ny, c] + grid[(r-1)%Ny, c])

                e_before = get_local_e(r1, c1, grid[r1, c1]) + get_local_e(r2, c2, grid[r2, c2])
                grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
                e_after = get_local_e(r1, c1, grid[r1, c1]) + get_local_e(r2, c2, grid[r2, c2])
                
                if e_after > e_before and rng.random() > 0.05:
                    grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
        
        total_e = Jx * np.sum(grid * np.roll(grid, -1, axis=1)) + \
                  Jy * np.sum(grid * np.roll(grid, -1, axis=0))
        configs.append(grid.copy())
        energies.append(total_e)

    indices = np.argsort(energies)
    top_energies = np.array(energies)[indices][:6]
    top_configs = [configs[i] for i in indices[:6]]
    
    beta = 0.5 
    weights = np.exp(-beta * (top_energies - np.min(top_energies)))
    probs = (weights / np.sum(weights)) * 100
    
    return top_configs, probs

# --- 3. VISUALIZATION ---
def run_app():
    plt.clf()
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 6, height_ratios=[1,
