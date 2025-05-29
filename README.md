# SubDataPy

**SubDataPy is a Python toolkit for performing various subsampling strategies on large datasets, particularly for applications in materials science simulations. It is specifically useful to subsample big datasets for the development of Machine Learning Interatomic Potentials. Subsampling methods have been shown to not only increase training speed by reducing the dataset but also potentially increase testing accuracy of the models by removing redundant data.**

## Key Features

* **Versatile Subsampling:** Implements several techniques:
    * Random Subsampling
    * Leverage Score Subsampling
    * Cook's Distance Subsampling (including iterative/one-step and stepwise approaches)
* **Configuration Aware:** Supports "block" versions of subsampling methods, considering groups of data rows (e.g., energy and forces for an atomic configuration) as single entities.
* **Data Handling:** Built on `numpy` and `cupynumeric` for efficient data manipulation.

## Installation

Currently, SubDataPy is under active development. To install it locally for development:

1.  Clone the repository:
    ```bash
    git clone [https://github.com/ilgarbaghishov/subdatapy.git](https://github.com/ilgarbaghishov/subdatapy.git) # Replace with your actual repo URL if different
    cd subdatapy
    ```
2.  Install the package in editable mode with test dependencies:
    ```bash
    pip install -e .[test]
    ```
    If you only need the runtime dependencies:
    ```bash
    pip install -e .
    ```

## Basic Usage

Here's a very basic example of how to use `RandomSubSampler` with SubDataPy:

```python
import numpy as np
import pandas as pd
from subdatapy.subsampler import RandomSubSampler

# 1. Create or load your data
# For example, small placeholder data:
X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.rand(100, 1)   # Target values
w = np.ones((100, 1))        # Weights
config_idxs = np.repeat(np.arange(20), 5) # 20 configurations, 5 rows each

# 2. Initialize the Subsampler
rs = RandomSubSampler(X, y=y, w=w, test_fraction=0.2, seed=42, config_idxs=config_idxs)


# 3. Create a subsample (this returns a mask, and also updates internal state)
# sub_mask = rs.create_subsample(subsample_fraction=0.5, seed=123)

# 4. Generate a DataFrame of errors for different fractions and repeats
errors_df = rs.create_subsample_errors_dataframe(
    subsample_fractions_list=[0.1, 0.25, 0.5],
    repeat_count_list=2, # Repeat 2 times for each fraction
    seed=43
)

print(errors_df.head())