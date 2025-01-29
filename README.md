<div align="center">
<img src="https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/blob/main/images/LIFDlogo.png"></a>
<a href="https://www.cemac.leeds.ac.uk/">
  <img src="https://github.com/cemac/cemac_generic/blob/master/Images/cemac.png"></a>
  <br>
</div>

# Leeds Institute for Fluid Dynamics Machine Learning For Earth Sciences

# Modal Decomposition for Reduced Order Modelling

[![GitHub release](https://img.shields.io/github/release/cemac/LIFD_ModalDecomposition.svg)](https://github.com/cemac/LIFD_ModalDecomposition/releases) [![GitHub top language](https://img.shields.io/github/languages/top/cemac/LIFD_ModalDecomposition.svg)](https://github.com/cemac/LIFD_ModalDecomposition) [![GitHub issues](https://img.shields.io/github/issues/cemac/LIFD_ModalDecomposition.svg)](https://github.com/cemac/LIFD_ModalDecomposition/issues) [![GitHub last commit](https://img.shields.io/github/last-commit/cemac/LIFD_ModalDecomposition.svg)](https://github.com/cemac/LIFD_ModalDecomposition/commits/main) [![GitHub All Releases](https://img.shields.io/github/downloads/cemac/LIFD_ModalDecomposition/total.svg)](https://github.com/cemac/LIFD_ModalDecomposition/releases) ![GitHub](https://img.shields.io/github/license/cemac/LIFD_DimensionalityReduction.svg) [![DOI](https://zenodo.org/badge/366734586.svg)](https://zenodo.org/badge/latestdoi/366734586)

[![LIFD_ENV_ML_NOTEBOOKS](https://github.com/cemac/LIFD_ModalDecomposition/actions/workflows/python-package-conda-md.yml/badge.svg)](https://github.com/cemac/LIFD_ModalDecomposition/actions/workflows/python-package-conda-md.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cemac/LIFD_ModalDecomposition/HEAD?labpath=Modal_Decomposition.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cemac/LIFD_ModalDecomposition/blob/main/Modal_Decomposition.ipynb)

This Jupyter notebook demonstrates how modal decomposition methods can be used for flow feature extraction in fluid mechanics datasets. Modal decomposition techniques, such as Proper Orthogonal Decomposition (POD) and Dynamic Mode Decomposition (DMD), help identify coherent structures in fluid flows, providing a useful dimension reduction for subsequent reduced order modelling. The example application focuses on the classic problem of fluid flow past a cylinder, showing how these methods can simplify complex flow fields into a manageable number of modes. This dimension reduction enables efficient and accurate reduced order modelling using Sparse Identification of Nonlinear Dynamics (SINDy).

## Quick look

### Quick start

**Binder and Colab buttons**

Will launch this tutorial in binder or Google Colab.

**Running locally**

If you're already familiar with Git, Anaconda and virtual environments, the environment you need to create is found in [MD.yml](https://github.com/cemac/LIFD_ModalDecomposition/blob/main/MD.yml) and the code below will install, activate and launch the notebook. The .yml file has been tested on the latest Linux, macOS and Windows operating systems.

```bash
git clone git@github.com:cemac/LIFD_ModalDecomposition.git
cd LIFD_ModalDecomposition
conda env create -f MD.yml
conda activate MD
jupyter-notebook
```

## Installation and requirements

This notebook is designed to run on a laptop with no special hardware required. It is recommended to do a local installation as outlined in the repository [howtorun](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/blob/main/howtorun.md) and [jupyter_notebooks](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/blob/main/jupyter_notebooks.md) sections. Otherwise, online compute platforms (e.g. Google Colab) are also supported.

## Licence Information

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">LIFD_ENV_ML_NOTEBOOKS</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://cemac.leeds.ac.uk/" property="cc:attributionName" rel="cc:attributionURL">CEMAC</a> are licenced under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

## Acknowledgements

Thanks to Calum Skene for providing code and material for this notebook. This tutorial is part of the [LIFD_ENV_ML_NOTEBOOKS](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS) series. Please refer to the parent repository for full acknowledgements.
