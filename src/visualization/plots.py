"""
Network filtering benchmark — visualization module.

Generates, per the results/ CSVs:
    - heatmap_jaccard.png             : mean Jaccard by network group x noise level (NBF)
    - heatmap_cosine_distance.png     : mean cosine distance by network group x noise level (NBF)
    - heatmap_filter_metric_variation.png : % variation per structural metric, by filter (NBF, reference noise level)
    - tables/ranking_expanded.csv/png : best filter per specific network type (e.g. watts_strogatz, biological, kitchen)
    - tables/ranking_grouped.csv/png  : best filter per macro group (Real nets / Simulated nets / Incident texts)
    - tables/naf_degenerate_cases.csv : (group, filter, noise_level) combos where NAF emptied the filtered graph

NBF (bench_noise_filtering) is treated as the primary benchmark for all figures/tables.
NAF (bench_structural_noise_filtering) is intentionally NOT charted — only its degenerate
cases are exported, to be cited as a footnote / discussed in text.

Usage:
    python -m src.visualization.plots --input results/ --output results/graphics
"""

import os
import glob
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as _cosine_dist


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
COSINE_METRICS = [m for m in STRUCTURAL_METRICS if m not in ('average_path_length', 'diameter')]

# Simulated-network model tokens matched against the (lowercased) filename.
# Order matters: check more specific tokens before shorter ones if ever ambiguous.
SIM_MODEL_TOKENS = [
    ('Barabasi_albert', 'barabasi_albert'),
    ('Watts_strogatz', 'watts_strogatz'),
    ('Grid', 'grid'),
    ('Random', 'random'),
    ('LFR', 'lfr'),
]

ICON_INFO_PATH = 'data/real_nets/ICON_info.csv'

GROUP_CLASS_REAL = 'Real nets'
GROUP_CLASS_SIM = 'Simulated nets'
GROUP_CLASS_INC = 'Incident texts'

NBF_BENCHMARK = 'bench_noise_filtering'
NAF_BENCHMARK = 'bench_structural_noise_filtering'

REFERENCE_NOISE_LEVEL = 0.3

# Column-name aliases across pipeline versions -> canonical name.
_COLUMN_ALIASES = {
    'calculate_jaccard_similarity': 'jaccard',
    'calculate_information_retention': 'information_retention',
}


# ---------------------------------------------------------------------------
# Loading and grouping
# ---------------------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename older/newer pipeline column names onto one canonical schema."""
    rename = {}
    for src, dst in _COLUMN_ALIASES.items():
        if src in df.columns and dst not in df.columns:
            rename[src] = dst
    if rename:
        df = df.rename(columns=rename)
    # If both aliases somehow ended up present, keep the canonical one.
    for src, dst in _COLUMN_ALIASES.items():
        if src in df.columns and dst in df.columns:
            df = df.drop(columns=[src])
    return df


def _match_sim_model(name_lower: str):
    for label, token in SIM_MODEL_TOKENS:
        if token in name_lower:
            return label
    return None


def _assign_group(df: pd.DataFrame, icon_info: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each row with:
        group       — specific network type (e.g. 'Watts_strogatz', 'Biological', 'kitchen')
        group_class — macro category (Real nets / Simulated nets / Incident texts)
    """
    df = df.copy()

    if 'graph_name' in df.columns:
        # New-format pipeline output (incident_texts): one graph == one category.
        df['group'] = df['graph_name']
        df['group_class'] = GROUP_CLASS_INC
        return df

    if 'filename' not in df.columns:
        df['group'] = None
        df['group_class'] = None
        return df

    fname_lower = df['filename'].astype(str).str.lower()
    is_real = df['filename'].astype(str).str.contains('_real_nets', regex=False)

    df['group'] = fname_lower.apply(_match_sim_model)
    df['group_class'] = GROUP_CLASS_SIM

    if is_real.any():
        clean_name = df.loc[is_real, 'filename'].astype(str).str.replace('_real_nets', '', regex=False)
        domain_lookup = icon_info.set_index('network_name')['networkDomain']
        df.loc[is_real, 'group'] = clean_name.map(domain_lookup).values
        df.loc[is_real, 'group_class'] = GROUP_CLASS_REAL

    return df


def load_all(input_path: str) -> pd.DataFrame:
    """
    Load every results CSV directly under input_path (non-recursive — this deliberately
    does NOT descend into an output/graphics subdirectory, so re-running the script never
    re-ingests its own generated tables as if they were benchmark results).
    """
    icon_info = pd.read_csv(ICON_INFO_PATH)
    frames = []

    for path in sorted(glob.glob(os.path.join(input_path, '*.csv'))):
        df = pd.read_csv(path)
        df = _normalize_columns(df)

        if 'graph_name' not in df.columns and 'filename' not in df.columns:
            print(f"[load] Skipping {os.path.basename(path)}: no graph_name/filename column.")
            continue

        df = _assign_group(df, icon_info)
        n_before = len(df)
        df = df.dropna(subset=['group'])
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"[load] {os.path.basename(path)}: dropped {n_dropped} row(s) with unrecognized group.")

        frames.append(df)

    if not frames:
        raise ValueError(f"No usable CSV files found directly under {input_path}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def _add_cosine_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'cosine_distance' column: cosine distance between the original and filtered
    structural-metric vectors (COSINE_METRICS only — excludes path-length/diameter, which
    are undefined for disconnected graphs). NaN where any needed column is missing/NaN or
    either vector is all-zero.

    Vectorized over the whole dataframe with numpy (not a per-row .apply) — this runs over
    hundreds of thousands of rows (10 filters x ~10k simulated graphs x 5 noise levels x 2
    benchmarks, plus real_nets/incident_texts), and a Python-level row loop there is slow
    enough to blow past a single command's time budget.
    """
    orig_cols = [f'{m}_original' for m in COSINE_METRICS]
    filt_cols = [f'{m}_filtered' for m in COSINE_METRICS]
    missing = [c for c in orig_cols + filt_cols if c not in df.columns]
    if missing:
        warnings.warn(f"Cosine distance: missing columns {missing}, filling as NaN.")
        for c in missing:
            df[c] = np.nan

    df = df.copy()
    a = df[orig_cols].to_numpy(dtype=float)
    b = df[filt_cols].to_numpy(dtype=float)

    dot = np.einsum('ij,ij->i', a, b)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    denom = norm_a * norm_b

    with np.errstate(invalid='ignore', divide='ignore'):
        cos_sim = np.where(denom > 0, dot / denom, np.nan)
    cos_dist = 1.0 - cos_sim

    invalid = np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1) | (denom == 0)
    cos_dist = np.where(invalid, np.nan, cos_dist)

    df['cosine_distance'] = cos_dist
    return df


# ---------------------------------------------------------------------------
# Heatmap: group x noise level, for a single metric (Jaccard, cosine distance, ...)
# ---------------------------------------------------------------------------

def plot_metric_heatmap(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_file: str,
    benchmark: str = NBF_BENCHMARK,
    cmap: str = 'viridis',
    vmin=None,
    vmax=None,
    fmt: str = '{:.2f}',
) -> None:
    """
    Rows = network group (every specific type: real-net domains, simulated models,
    incident_texts categories). Columns = noise level. Cell = mean value across all
    filters and all graphs in that group (filter-level detail lives in the ranking
    tables, not here). Annotated as 'mean' when only one sample contributes (e.g.
    incident_texts categories, which are a single graph — no std is defined), or
    'mean (+/-std)' when multiple graphs/instances contribute.
    """
    d = df[df['benchmark'] == benchmark]
    agg = d.groupby(['group', 'noise_level'])[metric].agg(['mean', 'std', 'count']).reset_index()

    groups = sorted(agg['group'].dropna().unique())
    noise_levels = sorted(agg['noise_level'].dropna().unique())

    mean_mat = np.full((len(groups), len(noise_levels)), np.nan)
    annot = np.full((len(groups), len(noise_levels)), '', dtype=object)

    for i, g in enumerate(groups):
        for j, nl in enumerate(noise_levels):
            row = agg[(agg['group'] == g) & (agg['noise_level'] == nl)]
            if row.empty:
                continue
            m, s, n = row.iloc[0][['mean', 'std', 'count']]
            if pd.isna(m):
                continue
            mean_mat[i, j] = m
            if n > 1 and not pd.isna(s):
                annot[i, j] = f"{fmt.format(m)}\n(+/-{fmt.format(s)})"
            else:
                annot[i, j] = fmt.format(m)

    fig_h = max(4.0, 0.32 * len(groups) + 1.5)
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    im = ax.imshow(mean_mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels([f'{nl:.1f}' for nl in noise_levels])
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_xlabel('Noise level')

    for i in range(len(groups)):
        for j in range(len(noise_levels)):
            if annot[i, j]:
                ax.text(j, i, annot[i, j], ha='center', va='center', fontsize=6)

    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"[heatmap:{metric}] {len(groups)} groups x {len(noise_levels)} noise levels -> {out_file}")


# ---------------------------------------------------------------------------
# Heatmap: filter x structural metric (% variation), at one reference noise level
# ---------------------------------------------------------------------------

def plot_filter_metric_heatmap(
    df: pd.DataFrame,
    out_file: str,
    noise_level: float = REFERENCE_NOISE_LEVEL,
    benchmark: str = NBF_BENCHMARK,
) -> None:
    """
    Rows = filter, columns = structural metric. Cell = mean percentage variation
    (filtered - original) / original, aggregated across ALL groups, at one reference
    noise level. 'N/A' where a filter has no usable data for that metric at all
    (e.g. it never produced a non-zero original value, or never ran in this dataset).
    """
    d = df[(df['benchmark'] == benchmark) & (np.isclose(df['noise_level'], noise_level))]
    filters = sorted(d['filter'].dropna().unique())

    mat = pd.DataFrame(index=filters, columns=STRUCTURAL_METRICS, dtype=float)

    for f in filters:
        fd = d[d['filter'] == f]
        for metric in STRUCTURAL_METRICS:
            oc, fc = f'{metric}_original', f'{metric}_filtered'
            if oc not in fd.columns or fc not in fd.columns:
                continue
            sub = fd[[oc, fc]].dropna()
            sub = sub[sub[oc] != 0]
            if sub.empty:
                continue
            pct = (sub[fc] - sub[oc]) / sub[oc]
            mat.loc[f, metric] = pct.mean() * 100

    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.45 * len(filters) + 1.5)))
    im = ax.imshow(mat.values.astype(float), aspect='auto', cmap='RdBu_r', vmin=-100, vmax=100)

    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            txt = 'N/A' if pd.isna(v) else f'{v:.0f}%'
            ax.text(j, i, txt, ha='center', va='center', fontsize=6)

    ax.set_title(f'% variation of structural metrics by filter (noise={noise_level}, {benchmark})', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[heatmap:filter x metric] {len(filters)} filters x {len(STRUCTURAL_METRICS)} metrics -> {out_file}")


# ---------------------------------------------------------------------------
# Ranking tables: best filter per group, by Jaccard (reference noise level)
# ---------------------------------------------------------------------------

def build_ranking_table(
    df: pd.DataFrame,
    key: str,
    noise_level: float = REFERENCE_NOISE_LEVEL,
    benchmark: str = NBF_BENCHMARK,
) -> pd.DataFrame:
    """
    One row per value of `key` ('group' for the expanded/per-type table, 'group_class'
    for the grouped/macro-category table): the filter with the highest mean Jaccard at
    the reference noise level, plus that same filter's Jaccard and cosine-distance values.
    """
    d = df[(df['benchmark'] == benchmark) & (np.isclose(df['noise_level'], noise_level))]

    rows = []
    for g, gd in d.groupby(key):
        by_filter = gd.groupby('filter').agg(
            jaccard=('jaccard', 'mean'),
            cosine_distance=('cosine_distance', 'mean'),
            n=('jaccard', 'count'),
        ).reset_index()
        by_filter = by_filter.dropna(subset=['jaccard'])
        if by_filter.empty:
            continue
        best = by_filter.sort_values('jaccard', ascending=False).iloc[0]
        rows.append({
            key: g,
            'best_filter': best['filter'],
            'jaccard': round(float(best['jaccard']), 3),
            'cosine_distance': (
                round(float(best['cosine_distance']), 3) if not pd.isna(best['cosine_distance']) else None
            ),
            'n_graphs': int(best['n']),
        })

    return pd.DataFrame(rows).sort_values(key).reset_index(drop=True)


def _save_table_png(table: pd.DataFrame, out_file: str, title: str) -> None:
    fig_h = max(1.5, 0.35 * len(table) + 1.0)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.axis('off')
    ax.set_title(title, fontsize=10, pad=10)
    tbl = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# NAF: degenerate-case report only (no chart — cited as footnote/text per plan)
# ---------------------------------------------------------------------------

def naf_degenerate_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    (group, filter, noise_level) combinations where NAF (bench_structural_noise_filtering)
    produced a filtered graph with zero edges (true_positive + false_positive == 0) — e.g.
    the disparity filter emptying the graph. Meant to back a footnote in the ranking table,
    not a chart.
    """
    d = df[df['benchmark'] == NAF_BENCHMARK].copy()
    if 'true_positive' not in d.columns or 'false_positive' not in d.columns:
        return pd.DataFrame(columns=['group', 'group_class', 'filter', 'noise_level'])

    d['filtered_edges'] = d['true_positive'].fillna(0) + d['false_positive'].fillna(0)
    degenerate = d[d['filtered_edges'] == 0]
    cols = ['group', 'group_class', 'filter', 'noise_level']
    return degenerate[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_graphics(input_path: str, output_path: str) -> None:
    """
    Generate the current agreed set of Results visuals from the raw results CSVs.

    Safety: this function will DELETE and recreate `output_path` (the old graphics),
    but it refuses to run unless output_path's own directory name is exactly 'graphics'
    — it will never rmtree `results/` (or anything else) by mistake. It never opens any
    file in `results/` for writing; result CSVs are read-only inputs here.
    """
    out_basename = os.path.basename(os.path.normpath(output_path))
    if out_basename != 'graphics':
        raise ValueError(
            f"Refusing to wipe '{output_path}': output_path must be a directory named "
            f"'graphics' (got basename '{out_basename}'). This check exists so this "
            f"function can never be pointed at the results CSVs by mistake."
        )

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'tables'), exist_ok=True)

    df = load_all(input_path)
    df = _add_cosine_distance(df)

    print(f"[load] {len(df)} rows | groups: {sorted(df['group'].dropna().unique())}")

    # Heatmap 1 — Jaccard, RQ1
    plot_metric_heatmap(
        df, 'jaccard',
        'Mean Jaccard similarity by network group and noise level (NBF)',
        os.path.join(output_path, 'heatmap_jaccard.png'),
        cmap='viridis', vmin=0, vmax=1,
    )

    # Heatmap 2 — cosine distance, RQ2
    plot_metric_heatmap(
        df, 'cosine_distance',
        'Mean cosine distance (structural-metric vectors) by network group and noise level (NBF)',
        os.path.join(output_path, 'heatmap_cosine_distance.png'),
        cmap='viridis_r',
    )

    # Heatmap 3 — filter x structural metric, % variation, RQ2 diagnostic
    plot_filter_metric_heatmap(
        df,
        os.path.join(output_path, 'heatmap_filter_metric_variation.png'),
    )

    # Ranking tables — RQ3
    expanded = build_ranking_table(df, key='group')
    grouped = build_ranking_table(df, key='group_class')

    expanded.to_csv(os.path.join(output_path, 'tables', 'ranking_expanded.csv'), index=False)
    grouped.to_csv(os.path.join(output_path, 'tables', 'ranking_grouped.csv'), index=False)
    _save_table_png(expanded, os.path.join(output_path, 'tables', 'ranking_expanded.png'),
                     'Best filter per network type (NBF, Jaccard, noise=%.1f)' % REFERENCE_NOISE_LEVEL)
    _save_table_png(grouped, os.path.join(output_path, 'tables', 'ranking_grouped.png'),
                     'Best filter per macro group (NBF, Jaccard, noise=%.1f)' % REFERENCE_NOISE_LEVEL)

    # NAF degenerate cases — for the footnote, no chart
    degenerate = naf_degenerate_report(df)
    degenerate.to_csv(os.path.join(output_path, 'tables', 'naf_degenerate_cases.csv'), index=False)

    print(f"[done] {len(expanded)} rows in ranking_expanded, {len(grouped)} in ranking_grouped, "
          f"{len(degenerate)} NAF degenerate case(s) logged.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate benchmark visualization plots.')
    parser.add_argument('--input', required=True, help='Directory containing results CSVs (top level only)')
    parser.add_argument('--output', required=True, help="Directory to save plots (must be named 'graphics')")
    args = parser.parse_args()

    generate_graphics(input_path=args.input, output_path=args.output)
