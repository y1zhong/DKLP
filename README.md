# Code for 'Deep kernel learning based Gaussian processes for Bayesian image regression analysis'

## Maintainer
- **Yuan Zhong** ([Contact](mailto:ylzhong@umich.edu))
  - Department of Biostatistics, University of Michigan

## Overview
This repository contains the code and simulated datasets to reproduce the simulation studies in "Deep kernel learning based Gaussian processes for Bayesian image regression analysis". Results from real data analysis can be provided upon request.

## Repository Structure

### Code
The `code` folder contains implementation for three image regression models and helper functions.

- **`DKLP_NPR.py`**
  - A Python module with `NPR` class that conduct fully Bayesian inference with deep kernel learning process on nonparametric regression model.
  
- **`DKLP_IS.py`**
  - A Python module with `IS` class that conduct fully Bayesian inference with deep kernel learning process on image-on-scalar regression model.

- **`DKLP_SI.py`**
  - A Python module with `SI` class that conduct fully Bayesian inference with deep kernel learning process on scalar-on-image regression model.

- **`helper.py`**
  - Additional helper functions such as defining DNN class and Bayesian FDR control.

- **`example.py`**
  - Toy example to run DKLP on three image regression model on simulated datasets.

### Data
The `data` folder contains simulated datasets.

- **`NPR_data.py`**
  - Pickle file that contains one-dimensional nonparameteric regression data. 
  
- **`IS_data.py`**
  - Pickle file that contains two-dimensional image-on-scalar regression data, with five spatial varying coefficients. 

- **`SI_data.py`**
- Pickle file that contains two-dimensional scalar-on-image regression data, with sparse spatial varying coefficient. 

## Dependencies

The DKLP module is built on Python (Version 3.12.2). To install the necessary python packages, run the following code.

```bash
pip install -r requirements.txt
```
