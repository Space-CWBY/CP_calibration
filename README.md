# MPCR–EVPSC parameter identification framework

## Project overview
- This repository provides standalone Python scripts for family-aware identification of ΔEVPSC Voce hardening parameters for multi-pass caliber-rolled (MPCR) ZK60 Mg alloy.

- The workflow consists of four sequential stages

  - **Stage 1 search** performs bi-objective exploration of the parameter space using **f0 = physics loss** and **f1 = prior penalty**  
  - **Stage 1 evaluation** summarizes simulation outputs and extracts deformation-mechanism activity descriptors  
  - **Stage 2 clustering** identifies solution families in the activity-feature space using **CLR(activity) features + GMM**  
  - **Stage 2 polishing** refines representative solutions by improving the physics fit while preserving the cluster-specific activity signature through distance-locked CMA-ES

This public release contains only the code required to reproduce the computational workflow.  
Raw experimental data are not included and are available from the corresponding author upon reasonable request.

## Repository contents
```
stage1_biobjective_search.py
stage1_evaluate.py
stage2_clustering.py
stage2_polish.py
requirements.txt
README.md
```

## Method summary
- **Material/process**: multi-pass caliber-rolled ZK60 Mg alloy
- **Constitutive model**: ΔEVPSC
- **Parameters**: 16 Voce hardening parameters for four deformation mechanisms
- **Clustering space**: block-wise CLR(activity) feature space
- **Visualization**: UMAP is used for qualitative visualization only
- **Polishing objective**: physics improvement with cluster-distance regularization

## Usage
Each step is provided as a standalone Python script.

```bash
python stage1_biobjective_search.py --help
python stage1_evaluate.py --help
python stage2_clustering.py --help
python stage2_polish.py --help
```

A typical workflow is:

```bash
python stage1_biobjective_search.py ...
python stage1_evaluate.py ...
python stage2_clustering.py ...
python stage2_polish.py ...
```

## Installation
Python 3.10 or newer is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Notes
- **f0** denotes the physics loss.
- **f1** denotes the prior penalty.
- UMAP is used only for visualization and is not used for cluster definition or distance locking.
- The physics-only search code used during internal development is not included in this public release.

## Citation
If this code is used in academic work, please cite the associated publication  
(citation details to be added upon publication).
