#!/usr/bin/env python3
"""
Generate LaTeX-friendly data tables from evaluation results.
Output format suitable for pgfplots, pgfplotstable, and direct table inclusion.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def export_for_pgfplots(df: pd.DataFrame, output_path: Path, metric: str):
    """
    Export data in pgfplots-compatible format.

    Creates a CSV that can be directly read by pgfplots with \addplot table.
    """
    # Pivot table for the metric
    methods = df['method'].unique()
    difficulties = ['basic', 'medium', 'hard']

    # Create data for plotting
    data = []
    for difficulty in difficulties:
        row = {'difficulty': difficulty}
        for method in methods:
            subset = df[(df['method'] == method) & (df['difficulty'] == difficulty) & (df['success'] == True)]
            if len(subset) > 0:
                row[f'{method}_mean'] = subset[metric].mean()
                row[f'{method}_std'] = subset[metric].std()
            else:
                row[f'{method}_mean'] = np.nan
                row[f'{method}_std'] = np.nan
        data.append(row)

    result_df = pd.DataFrame(data)
    result_df.to_csv(output_path, index=False)
    print(f"✓ Exported pgfplots data: {output_path}")


def export_success_rates(df: pd.DataFrame, output_path: Path):
    """Export success rates for bar charts."""
    methods = df['method'].unique()
    difficulties = ['basic', 'medium', 'hard']

    data = []
    for difficulty in difficulties:
        row = {'difficulty': difficulty}
        for method in methods:
            subset = df[(df['method'] == method) & (df['difficulty'] == difficulty)]
            success_rate = (subset['success'].sum() / len(subset)) * 100 if len(subset) > 0 else 0
            row[method] = success_rate
        data.append(row)

    result_df = pd.DataFrame(data)
    result_df.to_csv(output_path, index=False)
    print(f"✓ Exported success rates: {output_path}")


def export_latex_table(summary_df: pd.DataFrame, output_path: Path):
    """
    Generate complete LaTeX table code.
    """
    latex_code = r"""\begin{table}[htbp]
\centering
\caption{Performance Comparison: RL Agent vs APF Baseline}
\label{tab:comparison}
\begin{tabular}{llcccc}
\toprule
\textbf{Method} & \textbf{Difficulty} & \textbf{Success Rate} & \textbf{Time (s)} & \textbf{Path (m)} & \textbf{Smoothness} \\
\midrule
"""

    for _, row in summary_df.iterrows():
        method = row['method'].replace('_', ' ').title()
        difficulty = row['difficulty'].capitalize()
        success_rate = f"{row['success_rate_%']:.1f}\\%"

        # Format with mean ± std
        if pd.notna(row['mean_time_s']):
            time_str = f"${row['mean_time_s']:.2f} \\pm {row['std_time_s']:.2f}$"
        else:
            time_str = "---"

        if pd.notna(row['mean_path_length_m']):
            path_str = f"${row['mean_path_length_m']:.2f} \\pm {row['std_path_length_m']:.2f}$"
        else:
            path_str = "---"

        if pd.notna(row['mean_smoothness']):
            smooth_str = f"${row['mean_smoothness']:.4f} \\pm {row['std_smoothness']:.4f}$"
        else:
            smooth_str = "---"

        latex_code += f"{method} & {difficulty} & {success_rate} & {time_str} & {path_str} & {smooth_str} \\\\\n"

    latex_code += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex_code)

    print(f"✓ Exported LaTeX table: {output_path}")


def export_latex_plots(results_dir: Path, output_dir: Path):
    """
    Generate LaTeX code for all plots using pgfplots.
    """
    plots_code = r"""\documentclass[border=5pt]{standalone}
\usepackage{pgfplots}
\usepackage{pgfplotstable}
\pgfplotsset{compat=1.18}

\begin{document}

% ============================================================
% Figure 1: Success Rate Comparison
% ============================================================
\begin{tikzpicture}
\begin{axis}[
    ybar,
    bar width=15pt,
    ylabel={Success Rate (\%)},
    xlabel={Difficulty Level},
    symbolic x coords={basic,medium,hard},
    xtick=data,
    xticklabels={Basic, Medium, Hard},
    legend style={at={(0.5,-0.2)}, anchor=north, legend columns=-1},
    ymin=0, ymax=105,
    width=12cm,
    height=8cm,
    grid=major,
    enlarge x limits=0.2,
]

% Read data from CSV
\addplot table[x=difficulty, y=rl_agent, col sep=comma] {success_rates.csv};
\addplot table[x=difficulty, y=apf, col sep=comma] {success_rates.csv};

\legend{RL Agent, APF}

\end{axis}
\end{tikzpicture}

% ============================================================
% Figure 2: Path Length Comparison
% ============================================================
\begin{tikzpicture}
\begin{axis}[
    ybar,
    bar width=15pt,
    ylabel={Path Length (m)},
    xlabel={Difficulty Level},
    symbolic x coords={basic,medium,hard},
    xtick=data,
    xticklabels={Basic, Medium, Hard},
    legend style={at={(0.5,-0.2)}, anchor=north, legend columns=-1},
    ymin=0,
    width=12cm,
    height=8cm,
    grid=major,
    enlarge x limits=0.2,
    error bars/.cd,
    y dir=both,
    y explicit,
]

% Read data from CSV
\addplot +[error bars/.cd, y dir=both, y explicit]
    table[x=difficulty, y=rl_agent_mean, y error=rl_agent_std, col sep=comma]
    {path_length_data.csv};

\addplot +[error bars/.cd, y dir=both, y explicit]
    table[x=difficulty, y=apf_mean, y error=apf_std, col sep=comma]
    {path_length_data.csv};

\legend{RL Agent, APF}

\end{axis}
\end{tikzpicture}

% ============================================================
% Figure 3: Completion Time Comparison
% ============================================================
\begin{tikzpicture}
\begin{axis}[
    ybar,
    bar width=15pt,
    ylabel={Completion Time (s)},
    xlabel={Difficulty Level},
    symbolic x coords={basic,medium,hard},
    xtick=data,
    xticklabels={Basic, Medium, Hard},
    legend style={at={(0.5,-0.2)}, anchor=north, legend columns=-1},
    ymin=0,
    width=12cm,
    height=8cm,
    grid=major,
    enlarge x limits=0.2,
    error bars/.cd,
    y dir=both,
    y explicit,
]

\addplot +[error bars/.cd, y dir=both, y explicit]
    table[x=difficulty, y=rl_agent_mean, y error=rl_agent_std, col sep=comma]
    {time_data.csv};

\addplot +[error bars/.cd, y dir=both, y explicit]
    table[x=difficulty, y=apf_mean, y error=apf_std, col sep=comma]
    {time_data.csv};

\legend{RL Agent, APF}

\end{axis}
\end{tikzpicture}

% ============================================================
% Figure 4: Smoothness Comparison
% ============================================================
\begin{tikzpicture}
\begin{axis}[
    ybar,
    bar width=15pt,
    ylabel={Path Smoothness (lower is better)},
    xlabel={Difficulty Level},
    symbolic x coords={basic,medium,hard},
    xtick=data,
    xticklabels={Basic, Medium, Hard},
    legend style={at={(0.5,-0.2)}, anchor=north, legend columns=-1},
    ymin=0,
    width=12cm,
    height=8cm,
    grid=major,
    enlarge x limits=0.2,
    error bars/.cd,
    y dir=both,
    y explicit,
]

\addplot +[error bars/.cd, y dir=both, y explicit]
    table[x=difficulty, y=rl_agent_mean, y error=rl_agent_std, col sep=comma]
    {smoothness_data.csv};

\addplot +[error bars/.cd, y dir=both, y explicit]
    table[x=difficulty, y=apf_mean, y error=apf_std, col sep=comma]
    {smoothness_data.csv};

\legend{RL Agent, APF}

\end{axis}
\end{tikzpicture}

\end{document}
"""

    plots_file = output_dir / 'latex_plots.tex'
    with open(plots_file, 'w') as f:
        f.write(plots_code)

    print(f"✓ Exported LaTeX plots code: {plots_file}")

    # Also create a simplified version for embedding in main document
    embed_code = r"""% Copy these into your report document (requires pgfplots package)

% Success Rate Figure
\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    bar width=15pt,
    ylabel={Success Rate (\%)},
    xlabel={Difficulty Level},
    symbolic x coords={basic,medium,hard},
    xtick=data,
    xticklabels={Basic, Medium, Hard},
    legend style={at={(0.5,-0.15)}, anchor=north, legend columns=-1},
    ymin=0, ymax=105,
    width=0.9\textwidth,
    height=6cm,
    grid=major,
    enlarge x limits=0.2,
]

\addplot table[x=difficulty, y=rl_agent, col sep=comma] {figs/success_rates.csv};
\addplot table[x=difficulty, y=apf, col sep=comma] {figs/success_rates.csv};

\legend{RL Agent, APF}
\end{axis}
\end{tikzpicture}
\caption{Success rate comparison across different difficulty levels.}
\label{fig:success_rate}
\end{figure}

% Path Length Figure
\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    bar width=15pt,
    ylabel={Path Length (m)},
    xlabel={Difficulty Level},
    symbolic x coords={basic,medium,hard},
    xtick=data,
    xticklabels={Basic, Medium, Hard},
    legend style={at={(0.5,-0.15)}, anchor=north, legend columns=-1},
    ymin=0,
    width=0.9\textwidth,
    height=6cm,
    grid=major,
    enlarge x limits=0.2,
    error bars/.cd, y dir=both, y explicit,
]

\addplot +[error bars/.cd, y dir=both, y explicit]
    table[x=difficulty, y=rl_agent_mean, y error=rl_agent_std, col sep=comma]
    {figs/path_length_data.csv};

\addplot +[error bars/.cd, y dir=both, y explicit]
    table[x=difficulty, y=apf_mean, y error=apf_std, col sep=comma]
    {figs/path_length_data.csv};

\legend{RL Agent, APF}
\end{axis}
\end{tikzpicture}
\caption{Path length comparison for successful runs (mean ± std).}
\label{fig:path_length}
\end{figure}

% Add similar blocks for time and smoothness...
"""

    embed_file = output_dir / 'latex_figures_embed.tex'
    with open(embed_file, 'w') as f:
        f.write(embed_code)

    print(f"✓ Exported embeddable LaTeX figures: {embed_file}")


def main():
    parser = argparse.ArgumentParser(description='Export evaluation results to LaTeX format')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='runs/final_evaluation/results',
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for LaTeX files (default: results_dir/latex)'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (results_dir / 'latex')

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EXPORTING TO LATEX FORMAT")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Load results
    detailed_csv = results_dir / 'detailed_results.csv'
    summary_csv = results_dir / 'summary_statistics.csv'

    if not detailed_csv.exists():
        print(f"ERROR: detailed_results.csv not found")
        return

    detailed_df = pd.read_csv(detailed_csv)
    summary_df = pd.read_csv(summary_csv) if summary_csv.exists() else None

    print(f"Loaded {len(detailed_df)} evaluation results\n")

    # Export various formats
    print("Exporting data files...")

    # Success rates
    export_success_rates(detailed_df, output_dir / 'success_rates.csv')

    # Path length
    export_for_pgfplots(detailed_df, output_dir / 'path_length_data.csv', 'path_length_m')

    # Time
    export_for_pgfplots(detailed_df, output_dir / 'time_data.csv', 'time_s')

    # Smoothness
    export_for_pgfplots(detailed_df, output_dir / 'smoothness_data.csv', 'path_smoothness')

    print()

    # Export LaTeX code
    if summary_df is not None:
        print("Generating LaTeX code...")
        export_latex_table(summary_df, output_dir / 'comparison_table.tex')

    export_latex_plots(results_dir, output_dir)

    print(f"\n{'='*80}")
    print(f"EXPORT COMPLETE!")
    print(f"{'='*80}")
    print(f"\nGenerated files in {output_dir}/:")
    print(f"  - *.csv: Data files for pgfplots")
    print(f"  - comparison_table.tex: Complete table (copy into report)")
    print(f"  - latex_plots.tex: Standalone plots document")
    print(f"  - latex_figures_embed.tex: Code to embed in your report")
    print(f"\nUsage:")
    print(f"  1. Copy CSV files to your report's figs/ directory")
    print(f"  2. Copy code from latex_figures_embed.tex into your report")
    print(f"  3. Make sure you have \\usepackage{{pgfplots}} in preamble")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
