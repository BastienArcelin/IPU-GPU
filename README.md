# IPU-GPU

Performances comparison IPU (Graphcore) vs GPU.

Benchmark ideas (https://www.graphcore.ai/benchmarks): 
- classical deep neural network : training and inference
- Bayesian deep neaural network : training and inference

The images are generated with GalSim (https://github.com/GalSim-developers/GalSim, doc: http://galsim-developers.github.io/GalSim/_build/html/index.html) from parametric models fitted to real galaxies from the HST COSMOS catalog (which can be found from here: https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data).

## Installation
1. Clone the repository
```
git clone https://github.com/BastienArcelin/IPU-GPU
cd IPU-GPU
```
2. Install 
- with [conda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)
  - Install non-tensorflow packages with conda and activate the environnement
  ```
  conda env create -f ressources/env_gpu.yml
  conda activate env_gpu_benchmark
  ```
  - Install tensorflow packages with pip
  ```
  pip install tensorflow-gpu==2.1.0
  pip install tensorflow-probability==0.9.0
  ```
- with pip
  ```
  python3 -m pip install -r ressources/requirements.txt
  ```

## Before starting
1. Add a ```BENCHMARK_DATA``` environment variable, in the shell you are running, which points to the directory where you want your data to be stored.

Example, add to your ```.bashrc```:

```
export BENCHMARK_DATA='/path/to/data'
```

[//]: <>2. You need to download the COSMOS catalog. You can find it [here](https://zenodo.org/record/3242143#.Xv2pTvLgq9Y). You can chose the 
[//]: <>```COSMOS_25.2_training_sample.tar.gz``` (4.4 GB).
3. Save this file in the directory chosen for storing data (i.e. at ```BENCHMARK_DATA```).


## Notebook
A notebook where the benchmarck's results are visualized can be found [here](https://github.com/BastienArcelin/IPU-GPU/tree/master/notebooks)

## List of required packages
- pandas
- seaborn
- matplotlib
- numpy
- jupyter
- tensorflow-gpu==2.1.0
- tensorflow-probability==0.9.0

## Author
Bastien Arcelin - arcelin *at* apc.in2p3.fr
