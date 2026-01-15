
## Usage

The workflow for each model/panel consists of four steps:

1. Generate panel data: `generate_panel.py`
2. Calculate moments: `calculate_moments.py`
3. Compute Fama-French and Fama-MacBeth factors: `run_fama.py`
4. Compute DKKM factors (for all configured feature counts): `run_dkkm.py`

Each step generates a dictionary saved as a pickle file in the `outputs` directory.  The pickle files are read in subsequent steps. The outputs do not contain a panel index column.  Instead, the panel index is expected to be saved as a part of filenames.  Separate pickle files are generated for each panel with names `{model}_{index}_panel.pkl`, `{model}_{index}_moments.pkl`, `{model}_{index}_fama.pkl`, etc.

There are separate `utils` directories containing scripts for the three models.  There is also a `utils_factors` directory containing scripts used in `run_fama` and `run_dkkm`.  The original computation code, like sdf_compute, can be found in these utils directories.

The entire workflow can be run for one or more panels of a model using 

```
python main.py {model} {start_index} {end_index}
```

where the usual python convention of counting up to but not including `end_index` is followed.  For example,

```
python main.py kp14 0 5
```
creates five panels for the kp14 model indexed as 0, 1, 2, 3, 4.  `main.py` logs progress to `logs\{model}_{start_index}_{end_index}.log.  There are flags in config.py for the amount of data that is kept: keep_panel, keep_moments, and keep_factor_details.  If all flags are set to False, then only the data needed to compute Sharpe ratios and HJ distances is kept.

Rather than running `main.py`, the four steps can be executed individually:

1.  `python generate_panel.py {model} {n}` generates a panel with index n for the specified model.
2.  `python calculate_moments.py {model}_{n} reads  {model}_{n}_panel.pkl and calculates moments.
3.  `python run_fama {model_n}` reads {model}_{n}_panel.pkl and {model}_{n}_moments.pkl and generates Fama-French and Fama-MacBeth factors.
4.  `python run_dkkm {model_n} {k}` reads {model}_{n}_panel.pkl and {model}_{n}_moments.pkl and generates k DKKM factors.

There are two additional scripts in the root folder that are not employed in the main workflow.  analysis.py produces figures and tables.  view_pickle.py is a small utility script to describe the contents of a pickle file.

## Implementation Details

1. Ridge regression uses the kernel trick for computational efficiency when the number of factors exceeds the sample size (360 months).
2. Some numba acceleration is used in `run_dkkm.py`
3. `calculate_moments.py` processes months in chunks (controlled by config.py parameter) and writes each chunk to disk before proceeding to the next chunk. This is to conserve RAM.  Within each chunk, months are processed in parallel (controlled by n_jobs in config.py).  At the end of the script, the chunks are read and compiled into a single pickle file.  The intermediate pickle files are deleted.
4.  If book value at $t-1$ is zero, then ROE and asset growth at $t$ are set to zero.  The number of firm/months with zero book value is computed for each panel and logged.  The default value of BURNIN was raised from 200 to 300 to reduce the frequency of zero book values.
5.  Parameters in config.py determine the type of factors computed (rank-standardized or non-rank-standardized, including market or not including market),  The defaults are to rank-standardize and to include the market.  This differs from the original code that calculated both rank-standardized and non-rank-standardized factors.

## Configuration

All parameters are centralized in [`config.py`](config.py):

### Panel Dimensions

```python
N = 1000        # Number of firms
T = 720         # Time periods (months, excluding burnin)
BGN_BURNIN = 300   # Burnin period for BGN
KP14_BURNIN = 300  # Burnin period for KP14
GS21_BURNIN = 300  # Burnin period for GS21
N_JOBS = 10     # Parallel jobs for moment computation
```

### DKKM Parameters

```python
N_DKKM_FEATURES_LIST = [6, 36, 360]  # RFF feature counts to test
DKKM_RANK_STANDARDIZE = True         # Rank-standardize characteristics
INCLUDE_MKT = True                   # Include market portfolio
```

### Ridge Parameters

```python
# Shrinkage penalties (Berk-Jameson regularization)
ALPHA_LST = [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]  # For BGN/KP14
ALPHA_LST_GS = [0, 0.0000001, ..., 0.1, 1]          # For GS21
```

### Model Parameters

Each model has calibrated parameters (lines 85-159 in `config.py`).

## Testing

The `tests/` directory contains comprehensive regression tests that validate the refactored code against original implementations.

### Test Structure

All tests follow a consistent pattern:
1. Generate data with current code using fixed seed for reproducibility
2. Generate data with original code using same seed
3. Compare outputs with appropriate numerical tolerances
4. Report detailed statistics and differences

Tests are designed to fail immediately if required data (original pickle files) are missing, rather than silently proceeding with incomplete comparisons.

### Panel Generation Tests

**test_panel_bgn.py**, **test_panel_kp14.py**, **test_panel_gs21.py**

These tests validate that panel generation matches the original implementation exactly:
- Compares all panel columns (firmid, month, xret, characteristics)
- Compares all internal arrays (factor loadings, shocks, latent variables)
- Uses very tight tolerance (rtol=1e-14) since exact match is expected
- Validates shape, data types, and statistical properties

The tests use `tests/test_utils/run_generate_panel.py` wrapper to ensure consistent seeding across test runs.

### Moment Calculation Tests

**test_moments_bgn.py**, **test_moments_kp14.py**, **test_moments_gs21.py**

These tests validate SDF conditional moment calculations:
- Tests all moment quantities: rp, cond_var, rp_over_var, rp_abs, cond_cov
- Compares month-by-month statistics and overall distributions
- Uses rtol=1e-10 to account for accumulated floating-point error from matrix operations
- Validates that moments are computed correctly for all firms and time periods

Moments are the foundation for all downstream factor computations, so these tests are critical.

### Factor Extraction Tests

**test_fama.py** - Fama-French cross-sectional (FFC) and Fama-MacBeth regression (FMR):
- Tests factor computation for all alpha (ridge penalty) values
- Compares factor means, standard deviations, Sharpe ratios, and HJ distances
- Validates both FFC and FMR methodologies
- Uses rtol=1e-10 for numerical comparison

**test_dkkm.py** - Deep Kernel Machine (DKKM) factors:
- Tests all configured feature counts (typically 6, 36, 360)
- Tests all alpha values for ridge regularization
- Compares random Fourier features, factor statistics, and portfolio performance
- Validates rank-standardization if enabled
- Uses rtol=1e-10 for comparison

### Master Test Runner

**run_all_tests.py** - Orchestrates all regression tests:
```bash
python tests/run_all_tests.py
```

Runs all tests in sequence:
1. Panel generation (BGN, KP14, GS21)
2. Moment calculation (BGN, KP14, GS21)
3. Fama factors
4. DKKM factors

Reports overall success/failure and identifies which tests passed or failed. Useful for validating the entire codebase after making changes.

### Test Utilities

**tests/test_utils/comparison.py** - Numerical comparison functions:
```python
assert_close(arr1, arr2, rtol=1e-10, atol=1e-12, name="")
assert_dataframes_equal(df1, df2, rtol=1e-10, atol=1e-12)
compute_summary_stats(data_dict, name="")
print_comparison_summary(stats1, stats2)
```

**tests/test_utils/run_generate_panel.py** - Wrapper for panel generation with fixed seed:
- Sets TEST_SEED=12345 before calling generate_panel.py
- Ensures reproducibility across test runs
- Used by all panel generation tests

**Test Configuration** - Uses config.py values:
- N=50 firms (small for fast testing)
- T=400 periods
- Burnin=300 months (BGN_BURNIN/KP14_BURNIN/GS21_BURNIN)
- Fixed seed=12345 for reproducibility

### Recommended Numerical Tolerances

Different operations require different precision:
- Panel generation: rtol=1e-14 (exact match expected, pure arithmetic)
- Moments: rtol=1e-10 (matrix inversions accumulate floating-point error)
- Fama/DKKM: rtol=1e-10 (ridge regression and matrix operations)

### Test Data Requirements

Tests expect original code outputs to exist in `tests/original_code/`:
- `{model}_test_panel.pkl` - Panel data from original code
- `{model}_test_moments.pkl` - Moments from original code
- `{model}_test_fama.pkl` - Fama factors from original code
- `{model}_test_dkkm_{nfeatures}.pkl` - DKKM factors for each feature count

If any required file is missing, the test will fail immediately with a clear error message rather than proceeding with incomplete validation.

## Analysis

The [`analysis.py`](analysis.py) script automatically discovers all panels for each model and produces comprehensive LaTeX tables and figures summarizing factor performance.

### Usage

```bash
python analysis.py
```

The script automatically:
1. Discovers all available panels by scanning for `{model}_{index}_fama.pkl` files
2. Loads and processes Fama and DKKM results for each discovered panel
3. Computes performance metrics (Sharpe ratios and Hansen-Jagannathan distances)
4. Aggregates results across panels
5. Generates LaTeX tables and boxplot figures
6. Compiles all results into a single PDF document

### Performance Metrics

For each factor method and configuration, the script computes:

**Sharpe Ratio**: For each month, computes sharpe = mean / stdev, then averages across months within each panel. The tables show the mean Sharpe ratio across all panels.

**Hansen-Jagannathan Distance (HJD)**: For each month, computes hjd_realized = (mean - xret)^2, then aggregates as panel_hjd = sqrt(mean(hjd_realized)) within each panel. The tables show the mean HJD across all panels.

These metrics follow the methodology from the original analysis notebook.

### Output Files

**Tables** (saved to `tables/`):
- `{model}_fama.tex` - Fama-French results (FFC and FMR, sharpe and hjd)
- `{model}_dkkm_sharpe.tex` - DKKM Sharpe ratios by alpha and num_factors
- `{model}_dkkm_hjd.tex` - DKKM HJ distances by alpha and num_factors
- `all_results.tex` - Master LaTeX document with table of contents
- `all_results.pdf` - Compiled PDF (requires pdflatex)

**Figures** (saved to `figures/`):
- `{model}_fama_sharpe_boxplot.pdf` - Distribution of Fama Sharpe ratios across panels
- `{model}_fama_hjd_boxplot.pdf` - Distribution of Fama HJ distances across panels
- `{model}_dkkm_sharpe_boxplot.pdf` - Distribution of DKKM Sharpe ratios across panels
- `{model}_dkkm_hjd_boxplot.pdf` - Distribution of DKKM HJ distances across panels

Total: 9 LaTeX tables + 12 PDF figures (4 per model x 3 models)

### Boxplot Behavior

The boxplot functions intelligently adapt to the number of available panels:
- **Multiple panels**: Displays full boxplots showing distribution (median, quartiles, outliers)
- **Single panel**: Plots data as scatter markers instead of degenerate boxplots
  - Sharpe plots use steelblue markers
  - HJD plots use indianred markers

This ensures the visualizations are meaningful regardless of how many panels have been generated.

### Table Format

Tables show results organized by:
- **Rows**: Ridge penalty values (alpha)
- **Columns**:
  - Fama: Method (FFC/FMR) x Metric (sharpe/hjd)
  - DKKM: Number of features (e.g., 6, 36, 360)
- **Values**: Mean across all discovered panels

All tables include LaTeX captions, labels for cross-referencing, and are formatted for direct inclusion in academic papers.

### PDF Compilation

If `pdflatex` is available, the script automatically compiles `all_results.pdf` with:
- Title page
- Table of contents
- Organized sections for each model (BGN, KP14, GS21)
- Subsections for each method (Fama, DKKM)
- Tables and figures on separate pages for clarity

If `pdflatex` is not installed, the `.tex` file is still generated and can be compiled manually.
