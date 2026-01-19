"""
Analysis script to produce LaTeX tables from bop-noipca results.

This script automatically discovers and analyzes all panels for each model (BGN, KP14, GS21),
following the recipe from analysis.ipynb:
- Compute sharpe = mean / stdev for each month
- Compute hjd_realized = (mean - xret)^2 for each month
- Aggregate by panel, then across panels

Creates 9 LaTeX tables (3 for each model):
1. Fama table (FFC/FMR sharpe and hjd)
2-3. DKKM sharpe and hjd tables

Usage:
    python analysis.py

Outputs:
    - LaTeX tables saved to tables/
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import List
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR

# Output directories
SCRIPT_DIR = Path(__file__).parent
TABLES_DIR = SCRIPT_DIR / "tables"
FIGURES_DIR = SCRIPT_DIR / "figures"

# Create output directories if they don't exist
TABLES_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Configuration
MODELS = ['bgn', 'kp14', 'gs21']


def discover_panels(model: str) -> List[tuple]:
    """
    Discover all available (timestamp, panel_idx) pairs for a given model.

    Supports two filename formats:
    - Old format: {model}_{panel_idx}_fama.pkl
    - New format: {model}_{timestamp}_{panel_idx}_fama.pkl

    Each (timestamp, panel_idx) pair is treated as a unique panel.

    Args:
        model: Model name ('bgn', 'kp14', 'gs21')

    Returns:
        Sorted list of (timestamp, panel_idx) tuples
    """
    panels = set()

    # Scan for fama files (most likely to exist)
    pattern = os.path.join(DATA_DIR, f"{model}_*_fama.pkl")
    for filepath in glob.glob(pattern):
        # Extract components from filename
        filename = Path(filepath).stem
        parts = filename.split('_')

        if len(parts) == 3:
            # Old format: model_panelid_fama.pkl
            if parts[1].isdigit():
                timestamp = 'default'
                panel_idx = int(parts[1])
                panels.add((timestamp, panel_idx))
        elif len(parts) == 5:
            # New format: model_timestamp1_timestamp2_panelid_fama.pkl
            # e.g., bgn_20260117_092453_0_fama.pkl
            timestamp = f"{parts[1]}_{parts[2]}"
            if parts[3].isdigit():
                panel_idx = int(parts[3])
                panels.add((timestamp, panel_idx))

    return sorted(list(panels))


def load_and_process_fama(model: str, panels: List[tuple]) -> pd.DataFrame:
    """
    Load Fama results and compute sharpe/hjd following the notebook recipe.

    For each month:
    - sharpe = mean / stdev
    - hjd_realized = (sdf_ret - xret)^2

    Args:
        model: Model name
        panels: List of (timestamp, panel_idx) tuples

    Returns DataFrame with panel-level aggregates.
    """
    all_data = []

    for panel_id, (timestamp, panel_idx) in enumerate(panels):
        # Construct filename based on format
        if timestamp == 'default':
            # Old format: model_n_fama.pkl
            filename = f"{model}_{panel_idx}_fama.pkl"
        else:
            # New format: model_timestamp_n_fama.pkl
            filename = f"{model}_{timestamp}_{panel_idx}_fama.pkl"

        fama_file = os.path.join(DATA_DIR, filename)

        if not os.path.exists(fama_file):
            continue

        # Load fama stats (now includes sdf_ret from run_fama.py)
        with open(fama_file, 'rb') as f:
            fama_data = pickle.load(f)

        fama_stats = fama_data.get('fama_stats')
        if fama_stats is None or len(fama_stats) == 0:
            continue

        # Compute sharpe and hjd_sq for each month
        fama_stats = fama_stats.copy()
        fama_stats['sharpe'] = fama_stats['mean'] / fama_stats['stdev']
        fama_stats['hjd_sq'] = (fama_stats['sdf_ret'] - fama_stats['xret'])**2
        # Use sequential panel_id (0, 1, 2, ...) for aggregation
        fama_stats['panel'] = panel_id

        all_data.append(fama_stats[['panel', 'month', 'method', 'alpha', 'sharpe', 'hjd_sq']])

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def load_and_process_dkkm(model: str, panels: List[tuple]) -> pd.DataFrame:
    """
    Load DKKM results and compute sharpe/hjd following the notebook recipe.

    Args:
        model: Model name
        panels: List of (timestamp, panel_idx) tuples

    Returns DataFrame with panel-level aggregates.
    """
    all_data = []

    for panel_id, (timestamp, panel_idx) in enumerate(panels):
        # Construct pattern based on format
        if timestamp == 'default':
            # Old format: model_n_dkkm_*.pkl
            pattern = os.path.join(DATA_DIR, f"{model}_{panel_idx}_dkkm_*.pkl")
        else:
            # New format: model_timestamp_n_dkkm_*.pkl
            pattern = os.path.join(DATA_DIR, f"{model}_{timestamp}_{panel_idx}_dkkm_*.pkl")

        for filepath in glob.glob(pattern):
            # Extract nfeatures from filename
            filename = Path(filepath).stem
            parts = filename.split('_')

            # Parse based on format
            if timestamp == 'default':
                # Old format: model_n_dkkm_features.pkl (4 parts)
                if len(parts) < 4:
                    continue
                try:
                    nfeatures = int(parts[3])
                except ValueError:
                    continue
            else:
                # New format: model_timestamp1_timestamp2_n_dkkm_features.pkl (6 parts)
                if len(parts) < 6:
                    continue
                try:
                    nfeatures = int(parts[5])
                except ValueError:
                    continue

            # Load DKKM stats (now includes sdf_ret from run_dkkm.py)
            with open(filepath, 'rb') as f:
                dkkm_data = pickle.load(f)

            dkkm_stats = dkkm_data.get('dkkm_stats')
            if dkkm_stats is None or len(dkkm_stats) == 0:
                continue

            # Compute sharpe and hjd_sq for each month
            dkkm_stats = dkkm_stats.copy()
            dkkm_stats['sharpe'] = dkkm_stats['mean'] / dkkm_stats['stdev']
            dkkm_stats['hjd_sq'] = (dkkm_stats['sdf_ret'] - dkkm_stats['xret'])**2
            # Use sequential panel_id (0, 1, 2, ...) for aggregation
            dkkm_stats['panel'] = panel_id
            dkkm_stats['num_factors'] = nfeatures

            all_data.append(dkkm_stats[['panel', 'month', 'alpha', 'num_factors', 'sharpe', 'hjd_sq']])

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def create_fama_table(fama_df: pd.DataFrame, model: str, output_path: str):
    """
    Create Fama table for a single model.

    Rows: alpha values
    Columns: (FFC, sharpe), (FMR, sharpe), (FFC, hjd), (FMR, hjd)
    Values: Mean across panels
    """
    if len(fama_df) == 0:
        print(f"  WARNING: No Fama data for {model}")
        return

    # Get unique alpha values
    alphas = sorted(fama_df['alpha'].unique())

    # Create table data
    table_data = []
    for alpha in alphas:
        alpha_data = fama_df[fama_df['alpha'] == alpha]

        row = {'alpha': alpha}

        # FFC sharpe: mean across panels of (mean across months)
        ffc_data = alpha_data[alpha_data['method'] == 'ff']
        if len(ffc_data) > 0:
            ffc_panel_sharpe = ffc_data.groupby('panel')['sharpe'].mean()
            row['FFC_sharpe'] = ffc_panel_sharpe.mean()
        else:
            row['FFC_sharpe'] = np.nan

        # FMR sharpe: mean across panels of (mean across months)
        fmr_data = alpha_data[alpha_data['method'] == 'fm']
        if len(fmr_data) > 0:
            fmr_panel_sharpe = fmr_data.groupby('panel')['sharpe'].mean()
            row['FMR_sharpe'] = fmr_panel_sharpe.mean()
        else:
            row['FMR_sharpe'] = np.nan

        # FFC hjd: mean across panels of sqrt(mean across months of hjd_sq)
        if len(ffc_data) > 0:
            ffc_panel_hjd = ffc_data.groupby('panel')['hjd_sq'].apply(lambda x: np.sqrt(x.mean()))
            row['FFC_hjd'] = ffc_panel_hjd.mean()
        else:
            row['FFC_hjd'] = np.nan

        # FMR hjd: mean across panels of sqrt(mean across months of hjd_sq)
        if len(fmr_data) > 0:
            fmr_panel_hjd = fmr_data.groupby('panel')['hjd_sq'].apply(lambda x: np.sqrt(x.mean()))
            row['FMR_hjd'] = fmr_panel_hjd.mean()
        else:
            row['FMR_hjd'] = np.nan

        table_data.append(row)

    df = pd.DataFrame(table_data)
    df = df.set_index('alpha')

    # Create LaTeX table
    latex = df.to_latex(float_format="%.4f", na_rep="--",
                        column_format='r' + 'r'*len(df.columns),
                        escape=False)

    # Add caption and label
    caption = f"""\\caption{{\\textbf{{Fama-French Performance - {model.upper()}}}.
    Rows show different $\\alpha$ values. Columns show mean Sharpe ratio and
    Hansen-Jagannathan distance across panels for FFC and FMR methods.}}
\\label{{tab:fama_{model}}}"""

    latex = latex.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + caption)
    latex = r'\begin{table}[h]' + '\n' + r'\centering' + '\n' + latex + '\n' + r'\end{table}'

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"  Saved {output_path}")


def create_dkkm_tables(dkkm_df: pd.DataFrame, model: str, sharpe_path: str, hjd_path: str):
    """
    Create DKKM sharpe and hjd tables for a single model.

    Rows: alpha values
    Columns: num_factors
    Values: Mean across panels
    """
    if len(dkkm_df) == 0:
        print(f"  WARNING: No DKKM data for {model}")
        return

    # Get unique alpha and num_factors values
    alphas = sorted(dkkm_df['alpha'].unique())
    num_factors_vals = sorted(dkkm_df['num_factors'].unique())

    # Create sharpe table: mean across panels of (mean across months)
    sharpe_table = []
    for alpha in alphas:
        row = {'alpha': alpha}
        alpha_data = dkkm_df[dkkm_df['alpha'] == alpha]
        for nf in num_factors_vals:
            nf_data = alpha_data[alpha_data['num_factors'] == nf]
            if len(nf_data) > 0:
                panel_sharpe = nf_data.groupby('panel')['sharpe'].mean()
                row[nf] = panel_sharpe.mean()
            else:
                row[nf] = np.nan
        sharpe_table.append(row)

    sharpe_df = pd.DataFrame(sharpe_table).set_index('alpha')

    # Create hjd table: mean across panels of sqrt(mean across months of hjd_sq)
    hjd_table = []
    for alpha in alphas:
        row = {'alpha': alpha}
        alpha_data = dkkm_df[dkkm_df['alpha'] == alpha]
        for nf in num_factors_vals:
            nf_data = alpha_data[alpha_data['num_factors'] == nf]
            if len(nf_data) > 0:
                panel_hjd = nf_data.groupby('panel')['hjd_sq'].apply(lambda x: np.sqrt(x.mean()))
                row[nf] = panel_hjd.mean()
            else:
                row[nf] = np.nan
        hjd_table.append(row)

    hjd_df = pd.DataFrame(hjd_table).set_index('alpha')

    # Save sharpe table
    latex = sharpe_df.to_latex(float_format="%.4f", na_rep="--",
                                column_format='r' + 'r'*len(sharpe_df.columns),
                                escape=False)
    caption = f"""\\caption{{\\textbf{{DKKM Sharpe Ratios - {model.upper()}}}.
    Rows show different $\\alpha$ values. Columns show number of DKKM factors.
    Values are mean Sharpe ratios across panels.}}
\\label{{tab:dkkm_sharpe_{model}}}"""
    latex = latex.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + caption)
    latex = r'\begin{table}[h]' + '\n' + r'\centering' + '\n' + latex + '\n' + r'\end{table}'

    with open(sharpe_path, 'w') as f:
        f.write(latex)
    print(f"  Saved {sharpe_path}")

    # Save hjd table
    latex = hjd_df.to_latex(float_format="%.4f", na_rep="--",
                             column_format='r' + 'r'*len(hjd_df.columns),
                             escape=False)
    caption = f"""\\caption{{\\textbf{{DKKM Hansen-Jagannathan Distances - {model.upper()}}}.
    Rows show different $\\alpha$ values. Columns show number of DKKM factors.
    Values are mean HJ distances across panels.}}
\\label{{tab:dkkm_hjd_{model}}}"""
    latex = latex.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + caption)
    latex = r'\begin{table}[h]' + '\n' + r'\centering' + '\n' + latex + '\n' + r'\end{table}'

    with open(hjd_path, 'w') as f:
        f.write(latex)
    print(f"  Saved {hjd_path}")


def create_fama_boxplots(fama_df: pd.DataFrame, model: str):
    """Create Fama boxplots for sharpe and hjd distributions across panels."""
    if len(fama_df) == 0:
        print(f"  WARNING: No Fama data for {model}")
        return

    # Get unique alphas
    alphas = sorted(fama_df['alpha'].unique())

    # Create sharpe boxplot
    fig, ax = plt.subplots(figsize=(12, 6))

    boxplot_data = []
    labels = []

    for alpha in alphas:
        for method, method_label in [('ff', 'FFC'), ('fm', 'FMR')]:
            data = fama_df[(fama_df['alpha'] == alpha) & (fama_df['method'] == method)]
            if len(data) > 0:
                # Compute Sharpe per panel: mean(sharpe)
                panel_sharpe = data.groupby('panel')['sharpe'].mean()
                boxplot_data.append(panel_sharpe.values)
                labels.append(f"{method_label}\n$\\alpha$={alpha:.1e}")

    # Check if all data groups have only 1 point
    all_single = all(len(d) == 1 for d in boxplot_data)

    if all_single:
        # Plot as points instead of boxplots
        positions = range(1, len(boxplot_data) + 1)
        values = [d[0] for d in boxplot_data]
        ax.plot(positions, values, 'o', markersize=8, color='steelblue', markerfacecolor='lightblue',
                markeredgewidth=1.5, markeredgecolor='steelblue')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
    else:
        bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        plt.xticks(rotation=45, ha='right')

    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title(f'{model.upper()} - Fama Sharpe Ratios (Distribution Across Panels)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    sharpe_path = FIGURES_DIR / f"{model}_fama_sharpe_boxplot.pdf"
    plt.savefig(sharpe_path, bbox_inches='tight')
    print(f"  Saved {sharpe_path}")
    plt.close()

    # Create hjd boxplot
    fig, ax = plt.subplots(figsize=(12, 6))

    boxplot_data = []
    labels = []

    for alpha in alphas:
        for method, method_label in [('ff', 'FFC'), ('fm', 'FMR')]:
            data = fama_df[(fama_df['alpha'] == alpha) & (fama_df['method'] == method)]
            if len(data) > 0:
                # Compute HJD per panel: sqrt(mean(hjd_sq))
                panel_hjd = data.groupby('panel')['hjd_sq'].apply(lambda x: np.sqrt(x.mean()))
                boxplot_data.append(panel_hjd.values)
                labels.append(f"{method_label}\n$\\alpha$={alpha:.1e}")

    # Check if all data groups have only 1 point
    all_single = all(len(d) == 1 for d in boxplot_data)

    if all_single:
        # Plot as points instead of boxplots
        positions = range(1, len(boxplot_data) + 1)
        values = [d[0] for d in boxplot_data]
        ax.plot(positions, values, 'o', markersize=8, color='indianred', markerfacecolor='lightcoral',
                markeredgewidth=1.5, markeredgecolor='indianred')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
    else:
        bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        plt.xticks(rotation=45, ha='right')

    ax.set_ylabel('Hansen-Jagannathan Distance', fontsize=12)
    ax.set_title(f'{model.upper()} - Fama HJ Distances (Distribution Across Panels)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    hjd_path = FIGURES_DIR / f"{model}_fama_hjd_boxplot.pdf"
    plt.savefig(hjd_path, bbox_inches='tight')
    print(f"  Saved {hjd_path}")
    plt.close()


def create_dkkm_boxplots(dkkm_df: pd.DataFrame, model: str):
    """Create DKKM boxplots for sharpe and hjd distributions across panels."""
    if len(dkkm_df) == 0:
        print(f"  WARNING: No DKKM data for {model}")
        return

    # Get unique alphas and num_factors
    alphas = sorted(dkkm_df['alpha'].unique())
    num_factors_vals = sorted(dkkm_df['num_factors'].unique())

    # Create sharpe boxplot
    fig, ax = plt.subplots(figsize=(14, 6))

    boxplot_data = []
    labels = []

    for alpha in alphas:
        for nf in num_factors_vals:
            data = dkkm_df[(dkkm_df['alpha'] == alpha) & (dkkm_df['num_factors'] == nf)]
            if len(data) > 0:
                # Compute Sharpe per panel: mean(sharpe)
                panel_sharpe = data.groupby('panel')['sharpe'].mean()
                boxplot_data.append(panel_sharpe.values)
                labels.append(f"$\\alpha$={alpha:.1e}\nn={nf}")

    # Check if all data groups have only 1 point
    all_single = all(len(d) == 1 for d in boxplot_data)

    if all_single:
        # Plot as points instead of boxplots
        positions = range(1, len(boxplot_data) + 1)
        values = [d[0] for d in boxplot_data]
        ax.plot(positions, values, 'o', markersize=8, color='steelblue', markerfacecolor='lightblue',
                markeredgewidth=1.5, markeredgecolor='steelblue')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    else:
        bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        plt.xticks(rotation=45, ha='right', fontsize=8)

    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title(f'{model.upper()} - DKKM Sharpe Ratios (Distribution Across Panels)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    sharpe_path = FIGURES_DIR / f"{model}_dkkm_sharpe_boxplot.pdf"
    plt.savefig(sharpe_path, bbox_inches='tight')
    print(f"  Saved {sharpe_path}")
    plt.close()

    # Create hjd boxplot
    fig, ax = plt.subplots(figsize=(14, 6))

    boxplot_data = []
    labels = []

    for alpha in alphas:
        for nf in num_factors_vals:
            data = dkkm_df[(dkkm_df['alpha'] == alpha) & (dkkm_df['num_factors'] == nf)]
            if len(data) > 0:
                # Compute HJD per panel: sqrt(mean(hjd_sq))
                panel_hjd = data.groupby('panel')['hjd_sq'].apply(lambda x: np.sqrt(x.mean()))
                boxplot_data.append(panel_hjd.values)
                labels.append(f"$\\alpha$={alpha:.1e}\nn={nf}")

    # Check if all data groups have only 1 point
    all_single = all(len(d) == 1 for d in boxplot_data)

    if all_single:
        # Plot as points instead of boxplots
        positions = range(1, len(boxplot_data) + 1)
        values = [d[0] for d in boxplot_data]
        ax.plot(positions, values, 'o', markersize=8, color='indianred', markerfacecolor='lightcoral',
                markeredgewidth=1.5, markeredgecolor='indianred')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    else:
        bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        plt.xticks(rotation=45, ha='right', fontsize=8)

    ax.set_ylabel('Hansen-Jagannathan Distance', fontsize=12)
    ax.set_title(f'{model.upper()} - DKKM HJ Distances (Distribution Across Panels)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    hjd_path = FIGURES_DIR / f"{model}_dkkm_hjd_boxplot.pdf"
    plt.savefig(hjd_path, bbox_inches='tight')
    print(f"  Saved {hjd_path}")
    plt.close()


def generate_pdf():
    """Generate a PDF containing all LaTeX tables and figures."""
    import subprocess

    # Create master LaTeX document
    master_tex = TABLES_DIR / "all_results.tex"

    with open(master_tex, 'w') as f:
        f.write(r"\documentclass[11pt]{article}" + "\n")
        f.write(r"\usepackage{booktabs}" + "\n")
        f.write(r"\usepackage{graphicx}" + "\n")
        f.write(r"\usepackage{geometry}" + "\n")
        f.write(r"\geometry{margin=1in}" + "\n")
        f.write(r"\begin{document}" + "\n")
        f.write("\n")
        f.write(r"\title{BOP-NoIPCA Factor Model Results}" + "\n")
        f.write(r"\author{}" + "\n")
        f.write(r"\date{\today}" + "\n")
        f.write(r"\maketitle" + "\n")
        f.write(r"\tableofcontents" + "\n")
        f.write(r"\clearpage" + "\n")
        f.write("\n")

        # Include all tables and figures for each model
        for model in MODELS:
            f.write(f"\n\\section{{{model.upper()} Model Results}}\n\n")

            # Fama results
            f.write(f"\\subsection{{Fama-French Results}}\n\n")

            # Fama table
            fama_file = f"{model}_fama.tex"
            if (TABLES_DIR / fama_file).exists():
                f.write(f"\\input{{{fama_file}}}\n")
                f.write("\\clearpage\n\n")

            # Fama figures
            fama_sharpe_fig = f"..{os.sep}figures{os.sep}{model}_fama_sharpe_boxplot.pdf"
            if (FIGURES_DIR / f"{model}_fama_sharpe_boxplot.pdf").exists():
                f.write("\\begin{figure}[h]\n")
                f.write("\\centering\n")
                f.write(f"\\includegraphics[width=0.9\\textwidth]{{{fama_sharpe_fig}}}\n")
                f.write(f"\\caption{{Fama Sharpe Ratio Distribution - {model.upper()}}}\n")
                f.write("\\end{figure}\n")
                f.write("\\clearpage\n\n")

            fama_hjd_fig = f"..{os.sep}figures{os.sep}{model}_fama_hjd_boxplot.pdf"
            if (FIGURES_DIR / f"{model}_fama_hjd_boxplot.pdf").exists():
                f.write("\\begin{figure}[h]\n")
                f.write("\\centering\n")
                f.write(f"\\includegraphics[width=0.9\\textwidth]{{{fama_hjd_fig}}}\n")
                f.write(f"\\caption{{Fama HJ Distance Distribution - {model.upper()}}}\n")
                f.write("\\end{figure}\n")
                f.write("\\clearpage\n\n")

            # DKKM results
            f.write(f"\\subsection{{DKKM Results}}\n\n")

            # DKKM tables
            dkkm_sharpe_file = f"{model}_dkkm_sharpe.tex"
            if (TABLES_DIR / dkkm_sharpe_file).exists():
                f.write(f"\\input{{{dkkm_sharpe_file}}}\n")
                f.write("\\clearpage\n\n")

            dkkm_hjd_file = f"{model}_dkkm_hjd.tex"
            if (TABLES_DIR / dkkm_hjd_file).exists():
                f.write(f"\\input{{{dkkm_hjd_file}}}\n")
                f.write("\\clearpage\n\n")

            # DKKM figures
            dkkm_sharpe_fig = f"..{os.sep}figures{os.sep}{model}_dkkm_sharpe_boxplot.pdf"
            if (FIGURES_DIR / f"{model}_dkkm_sharpe_boxplot.pdf").exists():
                f.write("\\begin{figure}[h]\n")
                f.write("\\centering\n")
                f.write(f"\\includegraphics[width=0.9\\textwidth]{{{dkkm_sharpe_fig}}}\n")
                f.write(f"\\caption{{DKKM Sharpe Ratio Distribution - {model.upper()}}}\n")
                f.write("\\end{figure}\n")
                f.write("\\clearpage\n\n")

            dkkm_hjd_fig = f"..{os.sep}figures{os.sep}{model}_dkkm_hjd_boxplot.pdf"
            if (FIGURES_DIR / f"{model}_dkkm_hjd_boxplot.pdf").exists():
                f.write("\\begin{figure}[h]\n")
                f.write("\\centering\n")
                f.write(f"\\includegraphics[width=0.9\\textwidth]{{{dkkm_hjd_fig}}}\n")
                f.write(f"\\caption{{DKKM HJ Distance Distribution - {model.upper()}}}\n")
                f.write("\\end{figure}\n")
                f.write("\\clearpage\n\n")

        f.write(r"\end{document}" + "\n")

    print(f"  Created master LaTeX file: {master_tex}")

    # Compile PDF using pdflatex (run twice for table of contents and references)
    try:
        for _ in range(2):
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', 'all_results.tex'],
                cwd=str(TABLES_DIR),
                capture_output=True,
                text=True,
                timeout=120
            )

        pdf_file = TABLES_DIR / "all_results.pdf"
        if pdf_file.exists():
            print(f"  PDF generated successfully: {pdf_file}")

            # Clean up auxiliary files
            for ext in ['.aux', '.log', '.out', '.toc']:
                aux_file = TABLES_DIR / f"all_results{ext}"
                if aux_file.exists():
                    aux_file.unlink()
        else:
            print(f"  WARNING: PDF generation failed (pdflatex may not be installed)")
            print(f"  You can manually compile {master_tex} with pdflatex")

    except FileNotFoundError:
        print(f"  WARNING: pdflatex not found - skipping PDF generation")
        print(f"  You can manually compile {master_tex} with pdflatex")
    except subprocess.TimeoutExpired:
        print(f"  WARNING: pdflatex timed out")
    except Exception as e:
        print(f"  WARNING: PDF generation failed: {e}")


def main():
    """Main analysis function."""
    print("="*70)
    print("ANALYSIS - Creating LaTeX Tables")
    print("="*70)
    print()

    for model in MODELS:
        print(f"Processing {model.upper()}...")

        # Discover all (timestamp, panel_idx) pairs
        panels = discover_panels(model)
        if not panels:
            print(f"  WARNING: No panels found for {model}")
            print()
            continue

        print(f"  Found {len(panels)} panels: {panels}")

        # Load and process data
        print("  Loading Fama data...")
        fama_df = load_and_process_fama(model, panels)

        print("  Loading DKKM data...")
        dkkm_df = load_and_process_dkkm(model, panels)

        # Create tables
        print("  Creating tables...")

        # Table 1: Fama
        fama_path = TABLES_DIR / f"{model}_fama.tex"
        create_fama_table(fama_df, model, str(fama_path))

        # Tables 2-3: DKKM
        dkkm_sharpe_path = TABLES_DIR / f"{model}_dkkm_sharpe.tex"
        dkkm_hjd_path = TABLES_DIR / f"{model}_dkkm_hjd.tex"
        create_dkkm_tables(dkkm_df, model, str(dkkm_sharpe_path), str(dkkm_hjd_path))

        # Create figures
        print("  Creating figures...")
        create_fama_boxplots(fama_df, model)
        create_dkkm_boxplots(dkkm_df, model)

        print()

    # Generate PDF with all tables
    print("Generating PDF...")
    generate_pdf()

    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print()
    print(f"Tables: {TABLES_DIR}/")
    print(f"Figures: {FIGURES_DIR}/")
    print(f"PDF: {TABLES_DIR / 'all_results.pdf'}")


if __name__ == "__main__":
    main()
