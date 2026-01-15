# run_dkkm.py Pickle File Contents

**File location**: `outputs/{panel_id}_dkkm_{nfeatures}.pkl`
**Created by**: `run_dkkm.py`
**Created at**: [run_dkkm.py:212-224](run_dkkm.py#L212-L224)

## Overview

The DKKM (Deep Kernel using Random Fourier Features) pickle file contains factor returns computed using random Fourier feature transformations of firm characteristics, along with portfolio statistics and the random weight matrices used.

## Items Included (11 total)

---

### 1. `'dkkm_factors'` - DKKM Factor Returns

- **Type**: `pd.DataFrame`
- **Shape**: `(T, 2*nfeatures)` where T = number of months
- **Columns**: Numbered features `['0', '1', '2', ..., str(2*nfeatures-1)]`
- **Index**: Month numbers

**Creation trace**:
1. [run_dkkm.py:72-91](run_dkkm.py#L72-L91): `generate_rff_panel()` function
   - Generates random weight matrix W: shape `(nfeatures/2, n_chars + (1 if bgn else 0))`
   - Applies random scaling from gamma grid: `W = gamma * W`
   - Calls `dkkm.factors()` to compute factor returns
2. [dkkm_functions.py:46-70](utils_factors/dkkm_functions.py#L46-L70): `factors()` function
   - Loops through each month in parallel (using `joblib.Parallel`)
   - For each month, calls `rff()` to compute random Fourier features
   - Computes factor returns: `weights.T @ returns`
   - Concatenates monthly results
3. [dkkm_functions.py:16-25](utils_factors/dkkm_functions.py#L16-L25): `rff()` function
   - Rank-standardizes firm characteristics
   - For BGN model, adds risk-free rate
   - Computes: `Z = W @ X.T` (linear projection)
   - Applies nonlinear transformations: `Z1 = sin(Z)`, `Z2 = cos(Z)`
   - Concatenates: `[Z1; Z2]` to get 2*nfeatures features
   - Returns rank-standardized and non-rank-standardized versions

**Contents**: Monthly returns for random Fourier feature factors

**Note**: The returned factors depend on `rank_standardize` flag:
- If `True`: Returns rank-standardized RFF returns (`f_rs`)
- If `False`: Returns non-rank-standardized RFF returns (`f_nors`)

---

### 2. `'dkkm_stats'` - Portfolio Statistics for DKKM Factors

- **Type**: `pd.DataFrame`
- **Shape**: `(n_months * n_alphas, 8)`
- **Columns**: `['month', 'matrix', 'alpha', 'include_mkt', 'stdev', 'mean', 'xret', 'hjd']`
- **Index**: Default range index

**Creation trace**:
1. [run_dkkm.py:185-199](run_dkkm.py#L185-L199): Called via `portfolio_stats.compute_dkkm_portfolio_stats()`
2. [portfolio_stats.py:235-354](utils_factors/portfolio_stats.py#L235-L354): `compute_dkkm_portfolio_stats()` function
   - Loads pre-computed SDF moments from `{panel_id}_moments.pkl`
   - Loops over months (from `start_month` to `end_month`)
   - Loops over alpha values (ridge penalties)
   - For each combination:
     - [portfolio_stats.py:298-304](utils_factors/portfolio_stats.py#L298-L304): Computes optimal portfolio of factors using `dkkm.mve_data()`
       - Uses past 360 months of factor returns
       - Ridge regression with optional market factor
       - Returns portfolio weights on factors
     - [portfolio_stats.py:306-316](utils_factors/portfolio_stats.py#L306-L316): Recomputes RFF loadings
       - Calls `dkkm.rff()` with same weight matrix W
       - Gets loadings for stocks at current month
     - [portfolio_stats.py:318-328](utils_factors/portfolio_stats.py#L318-L328): Computes portfolio weights on stocks
       - `weights_on_stocks = loadings @ port_of_factors`
       - Expands to full N-dimensional vector
     - [portfolio_stats.py:330-341](utils_factors/portfolio_stats.py#L330-L341): Computes statistics
       - `stdev`: √(w' Σ w) using pre-computed conditional variance
       - `mean`: w' μ using pre-computed risk premium
       - `xret`: w' r using realized returns
       - `hjd`: Hansen-Jagannathan distance

**Column descriptions**:
- `month`: Month number
- `matrix`: Random matrix index (0 for first/only matrix)
- `alpha`: Ridge penalty parameter (regularization strength)
- `include_mkt`: Whether market factor is included (True/False)
- `stdev`: Portfolio standard deviation (from conditional variance)
- `mean`: Portfolio expected return (from risk premium)
- `xret`: Portfolio realized excess return
- `hjd`: Hansen-Jagannathan distance (pricing error metric)

**Contents**: Performance metrics for portfolios of DKKM factors across different regularization levels

---

### 3. `'weights'` - Random Weight Matrix

- **Type**: `np.ndarray`
- **Shape**: `(nfeatures/2, n_chars + (1 if bgn else 0))`
- **Values**: Random normal draws scaled by gamma grid

**Creation trace**:
1. [run_dkkm.py:74-80](run_dkkm.py#L74-L80): Generated in `generate_rff_panel()`
   ```python
   W = np.random.normal(size=(int(nfeatures/2), len(CHARS) + (MODEL == 'bgn')))
   gamma = np.random.choice(gamma_grid, size=(int(nfeatures/2), 1))
   W = gamma * W
   ```
2. [config.py:96](config.py#L96): `gamma_grid = np.arange(0.5, 1.1, 0.1)`

**Contents**: Random projection matrix used to transform characteristics into random Fourier features

**Note**:
- For BGN model: includes risk-free rate column, so shape is `(nfeatures/2, 6)`
- For KP14/GS21: only characteristics, so shape is `(nfeatures/2, 5)` or `(nfeatures/2, 6)` for GS21
- Each row is scaled by a randomly chosen gamma value

---

### 4. `'nfeatures'` - Number of Features

- **Type**: `int`
- **Value**: Specified in command line (e.g., 6, 36, 360)

**Creation trace**:
- [run_dkkm.py:123](run_dkkm.py#L123): Parsed from command-line arguments

**Contents**: Number of random Fourier features (D parameter)

**Note**: Final factor returns have `2*nfeatures` columns due to sin/cos transformations

---

### 5. `'nmat'` - Number of Weight Matrices

- **Type**: `int`
- **Value**: From `config.NMAT` (default: 1)

**Creation trace**:
- [run_dkkm.py:217](run_dkkm.py#L217): Copied from `CONFIG['nmat']`
- [config.py:48](config.py#L48): `NMAT = 1`

**Contents**: Number of random weight matrices generated

**Note**: Currently only the first matrix is used for statistics and saved in `'weights'`

---

### 6. `'rank_standardize'` - Standardization Flag

- **Type**: `bool`
- **Value**: From `config.DKKM_RANK_STANDARDIZE` (default: True)

**Creation trace**:
- [run_dkkm.py:218](run_dkkm.py#L218): Copied from `CONFIG['dkkm_rank_standardize']`
- [config.py:50](config.py#L50): `DKKM_RANK_STANDARDIZE = True`

**Contents**: Whether rank-standardization was applied to RFF features

**Impact**:
- If `True`: `dkkm_factors` contains rank-standardized returns
- If `False`: `dkkm_factors` contains non-rank-standardized returns

---

### 7. `'panel_id'` - Panel Identifier

- **Type**: `str`
- **Value**: e.g., `'bgn_0'`, `'kp14_5'`, `'gs21_12'`

**Creation trace**:
- [run_dkkm.py:108](run_dkkm.py#L108): Parsed from command-line arguments via `factor_utils.parse_panel_arguments()`

**Contents**: Identifier linking this output to the source panel data

---

### 8. `'model'` - Model Name

- **Type**: `str`
- **Value**: `'bgn'`, `'kp14'`, or `'gs21'`

**Creation trace**:
- [run_dkkm.py:126](run_dkkm.py#L126): Extracted from CONFIG via `CONFIG['model']`

**Contents**: Model identifier

---

### 9. `'chars'` - Characteristic Names

- **Type**: `list`
- **Value**:
  - BGN/KP14: `['size', 'bm', 'agr', 'roe', 'mom']` (5 characteristics)
  - GS21: `['size', 'bm', 'agr', 'roe', 'mom', 'mkt_lev']` (6 characteristics)

**Creation trace**:
- [run_dkkm.py:127](run_dkkm.py#L127): Extracted from CONFIG via `CONFIG['chars']`
- Defined in [config.py:77-82](config.py#L77-L82)

**Contents**: List of firm characteristics used in RFF transformation

---

### 10. `'start'` - Start Month

- **Type**: `int`
- **Value**: First month after burnin (e.g., 300 for burnin=300)

**Creation trace**:
- [run_dkkm.py:133](run_dkkm.py#L133): Returned from `factor_utils.prepare_panel()`

**Contents**: First month of factor returns

---

### 11. `'end'` - End Month

- **Type**: `int`
- **Value**: Last month in panel (e.g., 699 for T=400, burnin=300)

**Creation trace**:
- [run_dkkm.py:133](run_dkkm.py#L133): Returned from `factor_utils.prepare_panel()`

**Contents**: Last month of factor returns

---

## Summary

The pickle file contains:
- **Factor returns** (1 DataFrame): DKKM random Fourier feature factors
- **Portfolio statistics** (1 DataFrame): Performance metrics for optimal portfolios
- **Weight matrix** (1 array): Random projection matrix W
- **Configuration** (3 items): nfeatures, nmat, rank_standardize
- **Metadata** (5 items): panel_id, model, chars, start, end

**Key dependencies**:
- Requires `{panel_id}_panel.pkl` (panel data)
- Requires `{panel_id}_moments.pkl` (pre-computed SDF moments for statistics)
- Uses 360-month rolling windows for portfolio optimization

## Random Fourier Features Methodology

DKKM uses Random Fourier Features (RFF) to approximate a kernel function:

1. **Linear projection**: `Z = W @ X.T` where W is random, X are characteristics
2. **Nonlinear transformation**: Apply sin and cos to get `[sin(Z); cos(Z)]`
3. **Result**: `2*nfeatures` features that approximate kernel evaluation
4. **Factor returns**: These features are used as portfolio weights

The random weight matrix W is scaled by gamma values from a grid to control the "bandwidth" of the kernel approximation.

## Usage Example

```python
import pickle
import numpy as np
import pandas as pd

# Load the pickle file
with open('outputs/bgn_0_dkkm_360.pkl', 'rb') as f:
    results = pickle.load(f)

# Access DKKM factor returns
dkkm_factors = results['dkkm_factors']  # (T, 720) for nfeatures=360

# Access portfolio statistics
dkkm_stats = results['dkkm_stats']

# Access random weight matrix
W = results['weights']  # (180, 6) for BGN with nfeatures=360

# Access configuration
nfeatures = results['nfeatures']  # 360
rank_std = results['rank_standardize']  # True/False

# Access metadata
panel_id = results['panel_id']
model = results['model']
chars = results['chars']
start_month = results['start']
end_month = results['end']

# Filter statistics for specific alpha
alpha_001_stats = dkkm_stats[dkkm_stats['alpha'] == 0.001]
```

## Related Files

- **Input**: `outputs/{panel_id}_panel.pkl` - Panel data
- **Input**: `outputs/{panel_id}_moments.pkl` - Pre-computed SDF moments
- **Script**: `run_dkkm.py` - Generates this pickle file
- **Config**: `config.py` - Configuration parameters
  - `NMAT`: Number of random matrices (default: 1)
  - `DKKM_RANK_STANDARDIZE`: Whether to rank-standardize (default: True)
  - `GAMMA_GRID`: Grid for random scaling (default: [0.5, 0.6, ..., 1.0])

## Notes

- The number of factors in the output is `2*nfeatures` due to sin/cos transformations
- Common values for nfeatures: 6, 36, 360 (configured in `config.N_DKKM_FEATURES_LIST`)
- The random weight matrix W is crucial for reproducibility - saved for this purpose
- BGN model includes risk-free rate in characteristics, others don't
- Rank-standardization is applied by default to reduce sensitivity to outliers
- Multiple weight matrices can be generated (NMAT), but only first is currently used for stats
