"""
Network filtering benchmark — visualization module.

Entry point:
    generate_graphics(input_path, output_path)

Individual plot functions can also be called directly if only specific
classes are needed.

Usage:
    python generate_graphics.py --input results/ --output results/graphics
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from scipy.spatial.distance import cosine


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class GraphType(Enum):
    RANDOM          = 'Random'
    GRID            = 'Grid'
    BARABASI_ALBERT = 'Barabasi_albert'
    WATTS_STROGATZ  = 'Watts_strogatz'
    LFR             = 'LFR'
    BIOLOGICAL      = 'Biological'
    ECONOMIC        = 'Economic'
    INFORMATION     = 'Informational'
    SOCIAL          = 'Social'
    TECHNOLOGICAL   = 'Technological'
    TRANSPORTATION  = 'Transportation'
    REAL_NETS       = 'Real_nets'


STRUCTURAL_METRICS = [
    'average_degree',
    'average_clustering',
    'average_path_length',
    'diameter',
    'average_betweenness',
    'average_closeness',
    'global_efficiency',
    'degree_assortativity',
    'density',
    'transitivity',
    'degree_variance',
    'maximum_degree',
]

# Metrics excluded from cosine distance — undefined for disconnected graphs
COSINE_METRICS = [
    m for m in STRUCTURAL_METRICS
    if m not in ('average_path_length', 'diameter')
]

GROUP_COLUMNS = ['class', 'noise_level', 'benchmark', 'filter']

ICON_INFO_PATH = 'data/real_nets/ICON_info.csv'


# ---------------------------------------------------------------------------
# Data loading and aggregation
# ---------------------------------------------------------------------------

def _load_and_aggregate(input_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all results CSVs from a directory (recursively) and compute mean and std
    grouped by class, noise_level, benchmark, and filter.

    Expected directory structure:
        input_path/
            realNetsResults.csv       — real networks
            <anything_else>.csv       — simulated networks

    Real networks are split by domain (Biological, Economic, etc.) using ICON_info.csv,
    and also aggregated as a single 'Real_nets' class.

    Simulated network filenames follow the pattern: {number}{class}{number},
    where class matches one of the GraphType enum values (case-insensitive).

    Args:
        input_path (str): Directory containing the results CSVs (searched recursively).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (df_mean, df_std)
    """
    graph_type_values = [gt.value for gt in GraphType]
    dfs = []

    for root, _, files in os.walk(input_path):
        for filename in files:
            if not filename.endswith('.csv'):
                continue

            filepath = os.path.join(root, filename)
            df = pd.read_csv(filepath)

            if filename == 'realNetsResults.csv':
                # Load domain info and merge
                icon_info = pd.read_csv(ICON_INFO_PATH)
                df['filename'] = df['filename'].str.replace('_real_nets', '', regex=False)
                df = df.merge(
                    icon_info[['network_name', 'networkDomain']],
                    left_on='filename',
                    right_on='network_name',
                    how='left',
                )
                df['class'] = df['networkDomain']
                df.drop(columns=['networkDomain', 'network_name'], inplace=True)

                # Also add a Real_nets aggregate group
                real_nets_group = df.copy()
                real_nets_group['class'] = 'Real_nets'
                dfs.append(real_nets_group)
            else:
                df['class'] = df['filename'].apply(
                    lambda name: next(
                        (gt for gt in graph_type_values if gt.lower() in name.lower()),
                        None,
                    )
                )

            dfs.append(df)

    if not dfs:
        raise ValueError(f"No CSV files found in {input_path}")

    combined = pd.concat(dfs, ignore_index=True)

    unknown = combined['class'].isna().sum()
    if unknown > 0:
        print(f"Warning: {unknown} rows had unrecognized class and were dropped.")
        combined = combined.dropna(subset=['class'])

    mean_df = combined.groupby(GROUP_COLUMNS).mean(numeric_only=True).reset_index()
    std_df  = combined.groupby(GROUP_COLUMNS).std(numeric_only=True).reset_index()

    return mean_df, std_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_types(df_mean: pd.DataFrame, df_std: pd.DataFrame, benchmark: str):
    """Yield (graph_type, type_mean, type_std) for each non-empty GraphType."""
    bm_mean = df_mean[df_mean['benchmark'] == benchmark]
    bm_std  = df_std[df_std['benchmark'] == benchmark]

    for graph_type in GraphType:
        tm = bm_mean[bm_mean['class'] == graph_type.value]
        ts = bm_std[bm_std['class'] == graph_type.value]
        if not tm.empty:
            yield graph_type, tm, ts


def _save(path: str, filename: str) -> None:
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Class 1: Jaccard score by noise level
# ---------------------------------------------------------------------------

def plot_jaccard(df_mean: pd.DataFrame, df_std: pd.DataFrame, output_path: str) -> None:
    """Plot Jaccard score vs noise level for each filter, benchmark, and graph type."""
    for benchmark in df_mean['benchmark'].unique():
        for graph_type, tm, ts in _iter_types(df_mean, df_std, benchmark):
            for filter_name in tm['filter'].unique():
                fm = tm[tm['filter'] == filter_name].drop_duplicates(subset=['noise_level', 'jaccard'])
                fs = ts[ts['filter'] == filter_name].drop_duplicates(subset=['noise_level', 'jaccard'])
                plt.errorbar(fm['noise_level'], fm['jaccard'], yerr=fs['jaccard'], fmt='-o', label=filter_name)

            plt.title(f'Jaccard Score by Noise Level - {benchmark}')
            plt.xlabel('Noise Level')
            plt.ylabel('Jaccard Score')
            plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
            plt.ylim(0, 1)
            _save(
                os.path.join(output_path, 'class_1', graph_type.value, benchmark),
                f'Jaccard_{graph_type.value}.png',
            )


# ---------------------------------------------------------------------------
# Class 2: Precision, Recall, F1-score by noise level
# ---------------------------------------------------------------------------

def plot_predictive(df_mean: pd.DataFrame, df_std: pd.DataFrame, output_path: str) -> None:
    """Plot Precision, Recall, and F1-score vs noise level for each filter, benchmark, and graph type."""
    for benchmark in df_mean['benchmark'].unique():
        for graph_type, tm, ts in _iter_types(df_mean, df_std, benchmark):
            for metric in ['precision', 'recall', 'f1_score']:
                for filter_name in tm['filter'].unique():
                    fm = tm[tm['filter'] == filter_name]
                    fs = ts[ts['filter'] == filter_name]
                    plt.errorbar(fm['noise_level'], fm[metric], yerr=fs[metric], fmt='-o', label=filter_name)

                plt.title(f'{metric.capitalize()} by Noise Level - {benchmark}')
                plt.xlabel('Noise Level')
                plt.ylabel(f'{metric.capitalize()} Score')
                plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
                plt.ylim(0, 1)
                _save(
                    os.path.join(output_path, 'class_2', graph_type.value, benchmark),
                    f'{metric}_{graph_type.value}.png',
                )


# ---------------------------------------------------------------------------
# Class 3: Cosine distance between original and filtered metric vectors
# ---------------------------------------------------------------------------

def plot_cosine_distance(df_mean: pd.DataFrame, df_std: pd.DataFrame, output_path: str) -> None:
    """
    Plot cosine distance between the original and filtered structural metric vectors
    vs noise level for each filter, benchmark, and graph type.

    Metrics that are structurally undefined for disconnected graphs (average_path_length,
    diameter) are excluded. Filters with any remaining NaN values are omitted from the
    plot with a warning printed to the terminal.
    """
    for benchmark in df_mean['benchmark'].unique():
        for graph_type, tm, ts in _iter_types(df_mean, df_std, benchmark):
            orig_cols = [f'{m}_original' for m in COSINE_METRICS if f'{m}_original' in tm.columns]
            filt_cols = [f'{m}_filtered' for m in COSINE_METRICS if f'{m}_filtered' in tm.columns]

            plotted = False

            for filter_name in tm['filter'].unique():
                fm = tm[tm['filter'] == filter_name]
                fs = ts[ts['filter'] == filter_name]

                # Skip filters with any NaN in the metric vectors
                if fm[orig_cols + filt_cols].isna().any().any():
                    print(
                        f"[cosine] Skipping '{filter_name}' for {graph_type.value} / {benchmark} "
                        f"— NaN values in metric vector."
                    )
                    continue

                orig_std = fs[orig_cols]
                filt_std = fs[filt_cols]

                def _partial_orig(row):
                    a, b = row[orig_cols].values.astype(float), row[filt_cols].values.astype(float)
                    na = np.linalg.norm(a)
                    if na == 0 or np.linalg.norm(b) == 0:
                        return np.zeros_like(a)
                    return (np.dot(b, na) - np.dot(a, np.dot(a, b) / na)) / (na ** 2 * np.linalg.norm(b))

                def _partial_filt(row):
                    a, b = row[orig_cols].values.astype(float), row[filt_cols].values.astype(float)
                    nb = np.linalg.norm(b)
                    if nb == 0 or np.linalg.norm(a) == 0:
                        return np.zeros_like(b)
                    return (np.dot(a, nb) - np.dot(b, np.dot(b, a) / nb)) / (nb ** 2 * np.linalg.norm(a))

                pd_orig  = fm.apply(_partial_orig, axis=1)
                pd_filt  = fm.apply(_partial_filt, axis=1)
                cd_mean  = fm.apply(lambda row: cosine(
                    row[orig_cols].values.astype(float),
                    row[filt_cols].values.astype(float),
                ), axis=1)
                cd_std = [
                    np.sqrt(
                        np.dot(pd_orig[k], orig_std.loc[k].values) ** 2 +
                        np.dot(pd_filt[k], filt_std.loc[k].values) ** 2
                    )
                    for k, _ in orig_std.iterrows()
                ]

                plt.errorbar(fm['noise_level'], cd_mean, yerr=cd_std, fmt='-o', label=filter_name)
                plotted = True

            if plotted:
                plt.title(f'Cosine Distance by Noise Level - {benchmark}')
                plt.xlabel('Noise Level')
                plt.ylabel('Cosine Distance')
                plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))

            _save(
                os.path.join(output_path, 'class_3', graph_type.value, benchmark),
                f'cosine_distance_{graph_type.value}.png',
            )


# ---------------------------------------------------------------------------
# Class 4: Percentage variation of structural metrics
# ---------------------------------------------------------------------------

def plot_metric_variation(df_mean: pd.DataFrame, df_std: pd.DataFrame, output_path: str) -> None:
    """
    Plot percentage variation (filtered - original) / original vs noise level
    for each structural metric, filter, benchmark, and graph type.
    """
    for benchmark in df_mean['benchmark'].unique():
        for graph_type, tm, ts in _iter_types(df_mean, df_std, benchmark):
            for metric in STRUCTURAL_METRICS:
                orig_col = f'{metric}_original'
                filt_col = f'{metric}_filtered'

                if orig_col not in tm.columns or filt_col not in tm.columns:
                    continue

                plt.figure()

                for filter_name in tm['filter'].unique():
                    fm = tm[tm['filter'] == filter_name]
                    fs = ts[ts['filter'] == filter_name]

                    orig_mean = fm[orig_col]
                    filt_mean = fm[filt_col]
                    orig_std  = fs[orig_col]
                    filt_std  = fs[filt_col]

                    variation_mean = (filt_mean - orig_mean) / orig_mean
                    variation_std  = np.sqrt(
                        (filt_std / orig_mean) ** 2 +
                        ((filt_mean * orig_std) / (orig_mean ** 2)) ** 2
                    )

                    plt.errorbar(fm['noise_level'], variation_mean, yerr=variation_std, fmt='-o', label=filter_name)

                plt.title(f'{metric} Variation by Percentage - {benchmark}')
                plt.xlabel('Noise Level')
                plt.ylabel('Percentage Change')
                plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
                _save(
                    os.path.join(output_path, 'class_4', graph_type.value, benchmark),
                    f'{metric}_variation_{graph_type.value}.png',
                )


# ---------------------------------------------------------------------------
# Class 5: Absolute filtered metric value by noise level
# ---------------------------------------------------------------------------

def plot_metric_absolute(df_mean: pd.DataFrame, df_std: pd.DataFrame, output_path: str) -> None:
    """
    Plot absolute filtered metric value vs noise level for each structural metric,
    filter, benchmark, and graph type. Prepends noise_level=0 using the original
    metric value as baseline.
    """
    for benchmark in df_mean['benchmark'].unique():
        for graph_type, tm, ts in _iter_types(df_mean, df_std, benchmark):
            for metric in STRUCTURAL_METRICS:
                orig_col = f'{metric}_original'
                filt_col = f'{metric}_filtered'

                if filt_col not in tm.columns:
                    continue

                for filter_name in tm['filter'].unique():
                    fm = tm[tm['filter'] == filter_name].copy()
                    fs = ts[ts['filter'] == filter_name].copy()

                    baseline_mean = pd.DataFrame({'noise_level': [0.0], filt_col: [fm[orig_col].iloc[0]]})
                    baseline_std  = pd.DataFrame({'noise_level': [0.0], filt_col: [fs[orig_col].iloc[0]]})

                    fm = pd.concat([baseline_mean, fm], ignore_index=True)
                    fs = pd.concat([baseline_std,  fs], ignore_index=True)

                    plt.errorbar(fm['noise_level'], fm[filt_col], yerr=fs[filt_col], fmt='-o', label=filter_name)

                plt.title(f'{metric.replace("_", " ").capitalize()} by Noise Level - {benchmark}')
                plt.xlabel('Noise Level')
                plt.ylabel(metric.replace('_', ' ').capitalize())
                plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
                _save(
                    os.path.join(output_path, 'class_5', graph_type.value, benchmark),
                    f'{metric}_{graph_type.value}.png',
                )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_graphics(input_path: str, output_path: str) -> None:
    """
    Generate all benchmark visualization classes from raw results.

    Reads all CSVs from input_path recursively, computes mean and std internally,
    and saves all plots under output_path.

    Args:
        input_path (str): Directory containing results CSVs.
        output_path (str): Directory where plots will be saved.
    """
    df_mean, df_std = _load_and_aggregate(input_path)

    plot_jaccard(df_mean, df_std, output_path)
    plot_predictive(df_mean, df_std, output_path)
    plot_cosine_distance(df_mean, df_std, output_path)
    plot_metric_variation(df_mean, df_std, output_path)
    plot_metric_absolute(df_mean, df_std, output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate benchmark visualization plots.')
    parser.add_argument('--input',  required=True, help='Directory containing results CSVs (searched recursively)')
    parser.add_argument('--output', required=True, help='Directory to save plots')
    args = parser.parse_args()

    generate_graphics(input_path=args.input, output_path=args.output)