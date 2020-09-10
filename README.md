# IPU-GPU

Performances comparison IPU (Graphcore) vs GPU.

Benchmark ideas (https://www.graphcore.ai/benchmarks): 
- classical deep neural network : training and inference
- Bayesian deep neaural network : training and inference

The images are generated with GalSim (https://github.com/GalSim-developers/GalSim, doc: http://galsim-developers.github.io/GalSim/_build/html/index.html) from parametric models fitted to real galaxies from the HST COSMOS catalog (which can be found from here: https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data).

## Required packages
- tensorflow : 2.0.0 (or tensorflow-gpu)
- tensorflow-probability: 0.8.0
- galsim
