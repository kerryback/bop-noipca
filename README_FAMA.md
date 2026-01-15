# run_fama.py Pickle File Contents

**File location**: `outputs/{panel_id}_fama.pkl`
**Created by**: `run_fama.py`
**Created at**: [run_fama.py:168-180](run_fama.py#L168-L180)

## Overview

The Fama pickle file contains factor returns and portfolio statistics for Fama-French and Fama-MacBeth factors, along with model-implied factors (Taylor and Projection methods) extracted from the SDF computation.

## Items Included (11 total)

---

### 1. `'ff_returns'` - Fama-French Factor Returns

- **Type**: `pd.DataFrame`
- **Shape**: `(T, 6)` where T = number of months, 6 factors
- **Columns**: `['smb', 'hml', 'cma', 'rmw', 'umd', 'mkt_rf']`
- **Index**: Month numbers

**Creation trace**:
1. [run_fama.py:53-54](run_fama.py#L53-L54): Called via `fama.factors(fama.fama_french, panel, ...)`
2. [fama_functions.py:181-225](utils_factors/fama_functions.py#L181-L225): `factors()` function
   - Loops through each month in parallel (using `joblib.Parallel`)
   - For each month, calls `fama_french()` to compute factor weights
   - Computes factor returns: `weights.T @ returns`
   - Concatenates monthly results into DataFrame
3. [fama_functions.py:26-113](utils_factors/fama_functions.py#L26-L113): `fama_french()` function
   - Creates 2x3 sorts on size and each characteristic
   - Builds 6 portfolios (big/small × high/med/low)
   - Value-weights portfolios by market equity
   - Constructs long-short factors (high minus low)
   - Special handling: SMB from book-to-market terciles, CMA sign flip
   - Returns weights matrix for all stocks

**Contents**: Monthly returns for Fama-French style characteristic-sorted factors

---

### 2. `'fm_returns'` - Fama-MacBeth Factor Returns

- **Type**: `pd.DataFrame`
- **Shape**: `(T, 6)` where T = number of months, 6 factors
- **Columns**: `['smb', 'hml', 'cma', 'rmw', 'umd', 'mkt_rf']`
- **Index**: Month numbers

**Creation trace**:
1. [run_fama.py:55-57](run_fama.py#L55-L57): Called via `fama.factors(fama.fama_macbeth, panel, ...)`
2. [fama_functions.py:181-225](utils_factors/fama_functions.py#L181-L225): Same `factors()` function as above
   - Parallel computation across months
   - Calls `fama_macbeth()` instead of `fama_french()`
3. [fama_functions.py:116-178](utils_factors/fama_functions.py#L116-L178): `fama_macbeth()` function
   - Optionally standardizes characteristics (if `stdz_fm=True`)
   - Adds constant term
   - Computes pseudo-inverse projection matrix: `P = X @ pinv(X.T @ X)`
   - Removes intercept column
   - Normalizes weights to sum to 2 in absolute value
   - Returns weights matrix

**Contents**: Monthly returns for Fama-MacBeth cross-sectional regression factors

---

### 3. `'fama_stats'` - Portfolio Statistics for Fama Factors

- **Type**: `pd.DataFrame`
- **Shape**: `(n_months * n_methods * n_alphas, 7)`
- **Columns**: `['month', 'method', 'alpha', 'stdev', 'mean', 'xret', 'hjd']`
- **Index**: Default range index

**Creation trace**:
1. [run_fama.py:144-153](run_fama.py#L144-L153): Called via `portfolio_stats.compute_fama_portfolio_stats()`
2. [portfolio_stats.py:119-232](utils_factors/portfolio_stats.py#L119-L232): `compute_fama_portfolio_stats()` function
   - Loads pre-computed SDF moments from `{panel_id}_moments.pkl`
   - Loops over months (from `start_month` to `end_month`)
   - Loops over methods: `'ff'` and `'fm'`
   - Loops over alpha values (ridge penalties)
   - For each combination:
     - [portfolio_stats.py:183](utils_factors/portfolio_stats.py#L183): Computes optimal portfolio of factors using `fama.mve_data()`
       - Uses past 360 months of factor returns
       - Ridge regression: minimize ||1 - X'β||² + α||β||²
       - Returns portfolio weights on factors
     - [portfolio_stats.py:188-207](utils_factors/portfolio_stats.py#L188-L207): Gets factor loadings from panel
       - For FF: calls `fama.fama_french()` to recompute loadings
       - For FM: calls `fama.fama_macbeth()` to recompute loadings
     - [portfolio_stats.py:197-220](utils_factors/portfolio_stats.py#L197-L220): Computes portfolio weights on stocks
       - `weights_on_stocks = loadings @ port_of_factors`
       - Expands to full N-dimensional vector (fills with zeros for missing firms)
     - [portfolio_stats.py:209-220](utils_factors/portfolio_stats.py#L209-L220): Computes statistics
       - `stdev`: √(w' Σ w) using pre-computed conditional variance
       - `mean`: w' μ using pre-computed risk premium
       - `xret`: w' r using realized returns
       - `hjd`: Hansen-Jagannathan distance using pre-computed second moments

**Column descriptions**:
- `month`: Month number
- `method`: `'ff'` (Fama-French) or `'fm'` (Fama-MacBeth)
- `alpha`: Ridge penalty parameter (regularization strength)
- `stdev`: Portfolio standard deviation (from conditional variance)
- `mean`: Portfolio expected return (from risk premium)
- `xret`: Portfolio realized excess return
- `hjd`: Hansen-Jagannathan distance (pricing error metric)

**Contents**: Performance metrics (volatility, mean, excess return, HJ distance) for portfolios of Fama factors across different regularization levels

---

### 4. `'model_premia_taylor'` - Taylor Expansion Model Factors

- **Type**: `pd.DataFrame` or `None`
- **Shape**: `(T, n_chars)` if available
- **Columns**: Characteristic names (e.g., `['size', 'bm', 'agr', 'roe', 'mom']`)
- **Index**: Month numbers

**Creation trace**:
1. [run_fama.py:108-118](run_fama.py#L108-L118): Extracted from arrays data
   - Looks for keys starting with `'f_1_'` in arrays dictionary
   - Arrays come from `{panel_id}_panel.pkl` → `arrays_data['arrays']`
   - Created during panel generation by SDF computation modules
2. **Origin**: Generated by model-specific SDF modules:
   - BGN: `utils_bgn/sdf_compute.py` → Taylor expansion factors (f_1_)
   - KP14: `utils_kp14/sdf_compute_kp14.py` → Taylor expansion factors
   - GS21: `utils_gs21/sdf_compute_gs21.py` → Taylor expansion factors

**Contents**: Model-implied factor returns from first-order Taylor expansion of SDF

**Availability**: Present for all models (BGN, KP14, GS21)

---

### 5. `'model_premia_proj'` - Projection Model Factors

- **Type**: `pd.DataFrame` or `None`
- **Shape**: `(T, n_chars)` if available
- **Columns**: Characteristic names
- **Index**: Month numbers

**Creation trace**:
1. [run_fama.py:120-126](run_fama.py#L120-L126): Extracted from arrays data
   - Looks for keys starting with `'f_2_'` in arrays dictionary
   - Arrays come from `{panel_id}_panel.pkl`
2. **Origin**: Generated by model-specific SDF modules:
   - BGN: `utils_bgn/sdf_compute.py` → Projection factors (f_2_)
   - KP14: `utils_kp14/sdf_compute_kp14.py` → Projection factors
   - Note: GS21 only has `f_1_` (Taylor), no `f_2_` (Projection)

**Contents**: Model-implied factor returns from projection method

**Availability**: Present for BGN and KP14; `None` for GS21

---

### 6. `'model_stats'` - Portfolio Statistics for Model Factors

- **Type**: `pd.DataFrame` or `None`
- **Shape**: `(n_months * n_methods, 7)` if available
- **Columns**: `['month', 'method', 'alpha', 'stdev', 'mean', 'xret', 'hjd']`
- **Index**: Default range index

**Creation trace**:
1. [run_fama.py:138-141](run_fama.py#L138-L141): Called via `portfolio_stats.compute_model_portfolio_stats()`
2. [portfolio_stats.py:54-116](utils_factors/portfolio_stats.py#L54-L116): `compute_model_portfolio_stats()` function
   - Similar structure to `compute_fama_portfolio_stats()` but simpler
   - Uses `alpha=0` (no penalty) since model factors are already estimated
   - **Note**: Current implementation has placeholder statistics (many values are `np.nan`)
   - This function appears incomplete/simplified compared to other stats functions

**Contents**: Performance metrics for model-implied factors (Taylor and Projection methods)

**Status**: ⚠️ Partially implemented - contains some `np.nan` values

---

### 7. `'panel_id'` - Panel Identifier

- **Type**: `str`
- **Value**: e.g., `'bgn_0'`, `'kp14_5'`, `'gs21_12'`

**Creation trace**:
- [run_fama.py:71](run_fama.py#L71): Parsed from command-line arguments via `factor_utils.parse_panel_arguments()`

**Contents**: Identifier linking this output to the source panel data

---

### 8. `'model'` - Model Name

- **Type**: `str`
- **Value**: `'bgn'`, `'kp14'`, or `'gs21'`

**Creation trace**:
- [run_fama.py:77](run_fama.py#L77): Extracted from CONFIG via `CONFIG['model']`
- CONFIG loaded from `config.get_model_config(model_name)`

**Contents**: Model identifier

---

### 9. `'chars'` - Characteristic Names

- **Type**: `list`
- **Value**:
  - BGN/KP14: `['size', 'bm', 'agr', 'roe', 'mom']` (5 characteristics)
  - GS21: `['size', 'bm', 'agr', 'roe', 'mom', 'mkt_lev']` (6 characteristics)

**Creation trace**:
- [run_fama.py:78](run_fama.py#L78): Extracted from CONFIG via `CONFIG['chars']`
- Defined in [config.py:77-82](config.py#L77-L82)

**Contents**: List of firm characteristics used in factor construction

---

### 10. `'start'` - Start Month

- **Type**: `int`
- **Value**: First month after burnin (e.g., 300 for burnin=300)

**Creation trace**:
- [run_fama.py:85](run_fama.py#L85): Returned from `factor_utils.prepare_panel()`
- Equals `panel.index.get_level_values('month').min()`

**Contents**: First month of factor returns

---

### 11. `'end'` - End Month

- **Type**: `int`
- **Value**: Last month in panel (e.g., 699 for T=400, burnin=300)

**Creation trace**:
- [run_fama.py:85](run_fama.py#L85): Returned from `factor_utils.prepare_panel()`
- Equals `panel.index.get_level_values('month').max()`

**Contents**: Last month of factor returns

---

## Summary

The pickle file contains:
- **Factor returns** (2 DataFrames): FF and FM characteristic-based factors
- **Model factors** (2 DataFrames or None): Taylor and Projection SDF-implied factors
- **Portfolio statistics** (2 DataFrames): Performance metrics for optimal portfolios
- **Metadata** (5 items): panel_id, model, chars, start, end

**Key dependencies**:
- Requires `{panel_id}_panel.pkl` (panel data with arrays)
- Requires `{panel_id}_moments.pkl` (pre-computed SDF moments for statistics)
- Uses 360-month rolling windows for portfolio optimization

## Usage Example

```python
import pickle

# Load the pickle file
with open('outputs/bgn_0_fama.pkl', 'rb') as f:
    results = pickle.load(f)

# Access factor returns
ff_returns = results['ff_returns']  # Fama-French factors
fm_returns = results['fm_returns']  # Fama-MacBeth factors

# Access portfolio statistics
fama_stats = results['fama_stats']

# Access model-implied factors (if available)
taylor_factors = results['model_premia_taylor']
proj_factors = results['model_premia_proj']

# Access metadata
panel_id = results['panel_id']
model = results['model']
chars = results['chars']
start_month = results['start']
end_month = results['end']
```

## Related Files

- **Input**: `outputs/{panel_id}_panel.pkl` - Panel data with arrays
- **Input**: `outputs/{panel_id}_moments.pkl` - Pre-computed SDF moments
- **Script**: `run_fama.py` - Generates this pickle file
- **Config**: `config.py` - Configuration parameters

## Notes

- All factor returns are excess returns (relative to risk-free rate)
- Portfolio statistics use pre-computed SDF conditional moments for efficiency
- The 360-month rolling window is hardcoded for portfolio optimization
- Model factors (Taylor/Projection) come from the SDF computation, not Fama methods
- GS21 model only has Taylor factors, no Projection factors
