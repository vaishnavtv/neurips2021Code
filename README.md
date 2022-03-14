The code to generate data and plots for the paper are contained in this package. 
There are 4 sub-folders - one each for the baseline implementation of the Van der Pol and the Van der Pol Rayleigh oscillators, as well as one each for the OT-PINNs implementation. 

Please activate the relevant parent directory in Julia before running any file.


Julia 1.6.0 was used to perform simulations. Package dependencies:
- NeuralPDE
- ModelingToolkit
- Flux
- Symbolics
- JLD2 (for saving/loading data)
- GalacticOptim
- DiffEqFlux
- ForwardDiff
- LinearAlgebra
- Convex
- Mosek 
- MosekTools
- PyPlot
- Random
- Trapz 
- VectorizedRoutines
- Printf

Most of these packages are available open-source. 
Mosek licenses are available to academic faculty, students, or staff for free for research or educational purposes. 

This paper was submitted for consideration at Neurips 2021. Unfortunately, it did not get accepted for publication, but we believe the results are worthy and of interest to the PINNs community. 
The full paper is on [Arxiv](https://arxiv.org/abs/2105.12307).
