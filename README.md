# Companion code for [Masuda & Nunota (2026)]()

This repository contains the notebooks and data used for selected analyses in the paper:

* `analytic_jacobian.ipynb` reproduces the analytic Jacobian checks presented in **Section 2**.
* `fit_simulated_data.ipynb` reproduces the simulated-data fitting and comparison presented in **Section 4**.

## Repository contents

* `analytic_jacobian.ipynb`
  Notebook for validating the analytic Jacobian.

* `generate_simulated_data.ipynb`
  Notebook for generating the simulated data used in the benchmark analysis.

* `fit_simulated_data.ipynb`
  Notebook for fitting the simulated data and comparing sampling performance.

* `numpyro_sample_orbits.py`
  Python module containing the NumPyro model and orbit-sampling utilities used by the notebooks.

* `simulated_data.csv`
  Simulated astrometric data used in the fitting example.

* `simulated_data_params.npz`
  Parameters associated with the simulated data set.

* `hr8799_konopacky.txt`
  Input data file from [Konopacky et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016AJ....152...28K/abstract) used in constructing the simulated benchmark data.
