DMRG-style Visualizer for the 6x6 Heisenberg Model

This Streamlit application provides an interactive visualization of the S=1/2 Quantum Heisenberg Model on a 6×6 lattice. It is designed to illustrate how interaction anisotropy (Jx​,Jy​) and quantum constraints shape the magnetic ground state and its excitations.
1. Physical Model & Hamiltonian

The simulation solves for the physics of the Heisenberg Hamiltonian on a 2D square lattice with Periodic Boundary Conditions (PBC):
H=Jx​⟨i,j⟩x​∑​Si​⋅Sj​+Jy​⟨i,j⟩y​∑​Si​⋅Sj​

    Interaction Signs: * J<0: Ferromagnetic (favors parallel spins).

        J>0: Antiferromagnetic (favors anti-parallel spins).

    Geometry: A 6×6 torus (36 sites total).

2. Constraints & Symmetries

To remain physically consistent with the ground state of a quantum magnet in the singlet sector, the following are assumed:

    Stotalz​=0 Subspace: The simulation strictly maintains N↑​=N↓​=18. This reflects the most probable sector for the ground state of an antiferromagnet and the constraint often used in DMRG or Exact Diagonalization.

    Quantum Fluctuations: While the configurations shown are classical "snapshots" in the Sz basis, the ranking and correlations reflect the underlying quantum order where the true ground state is a superposition of many such states.

3. Interpreting the Results
Row 1: Most Likely Configurations & Probabilities

Quantum ground states are not static; they are superpositions of basis states.

    Ranking: Configurations are generated via a Metropolis-style exchange and ranked by their classical energy relative to the Hamiltonian.

    Probabilities: Calculated using a Boltzmann-like weighting P∝e−βE to estimate the relative importance of each configuration in the ground state wavefunction.

    Phases: You will observe Néel order (checkerboard) when both J are positive, and Striped order when Jx​ and Jy​ have opposite signs.

Row 2: Energy Spectrum (Anderson Tower of States)

This row visualizes the six lowest distinct energy levels.

    Logic: In a finite-size quantum magnet, the symmetry breaking of the infinite lattice is represented by a set of closely spaced levels known as the Anderson Tower of States.

    Degeneracy (g): Represents the multiplicity of the spin-multiplet (e.g., singlets, triplets, quintets). The lines represent the energies of these states as they appear in the Sz=0 sector.

Row 3: Spin-Spin Correlation Map C(i,j)

The heatmap shows the correlation function C(i,j)=⟨S0z​Si,jz​⟩.

    Fixed Range: The color bar is locked at [-0.25, 0.25].

    Interpretation: A value of +0.25 (dark red) indicates perfect alignment with the reference site, while −0.25 (dark blue) indicates perfect anti-alignment.

    Symmetry: Because of the torus geometry, the correlation is calculated relative to the (0,0) site and reflects the long-range order of the selected phase.

Installation & Deployment

    Requirements: streamlit, numpy, matplotlib.

    Run Locally: streamlit run app.py

    Deployment: Compatible with Streamlit Community Cloud.
