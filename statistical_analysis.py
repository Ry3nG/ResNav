#!/usr/bin/env python3
"""
Statistical analysis of evaluation results.
Computes t-tests, effect sizes, and confidence intervals.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0

    return (group1.mean() - group2.mean()) / pooled_std

def interpret_cohens_d(d):
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"

def confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for mean."""
    n = len(data)
    mean = data.mean()
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h

def analyze_metric(rl_data, apf_data, metric_name, higher_is_better=True):
    """Perform statistical analysis on a single metric."""
    print(f"\n{'='*60}")
    print(f"METRIC: {metric_name}")
    print(f"{'='*60}")

    # Basic statistics
    rl_mean, rl_std = rl_data.mean(), rl_data.std()
    apf_mean, apf_std = apf_data.mean(), apf_data.std()

    print(f"\nDescriptive Statistics:")
    print(f"  ResNav: {rl_mean:.4f} ± {rl_std:.4f} (n={len(rl_data)})")
    print(f"  APF:    {apf_mean:.4f} ± {apf_std:.4f} (n={len(apf_data)})")

    # Confidence intervals
    rl_ci = confidence_interval(rl_data)
    apf_ci = confidence_interval(apf_data)
    print(f"\n95% Confidence Intervals:")
    print(f"  ResNav: [{rl_ci[0]:.4f}, {rl_ci[1]:.4f}]")
    print(f"  APF:    [{apf_ci[0]:.4f}, {apf_ci[1]:.4f}]")

    # T-test (independent samples, two-tailed)
    t_stat, p_value = stats.ttest_ind(rl_data, apf_data)

    print(f"\nIndependent Samples T-Test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.001:
        sig_level = "*** (p < 0.001)"
    elif p_value < 0.01:
        sig_level = "** (p < 0.01)"
    elif p_value < 0.05:
        sig_level = "* (p < 0.05)"
    else:
        sig_level = "ns (not significant)"
    print(f"  Significance: {sig_level}")

    # Effect size
    d = cohens_d(rl_data, apf_data)
    interpretation = interpret_cohens_d(d)

    print(f"\nEffect Size (Cohen's d):")
    print(f"  d = {d:.4f} ({interpretation})")

    # Determine if result is favorable
    if higher_is_better:
        favorable = rl_mean > apf_mean
        diff_pct = ((rl_mean - apf_mean) / apf_mean) * 100 if apf_mean != 0 else 0
    else:
        favorable = rl_mean < apf_mean
        diff_pct = ((apf_mean - rl_mean) / apf_mean) * 100 if apf_mean != 0 else 0

    print(f"\nResult Summary:")
    if favorable and p_value < 0.05:
        print(f"  ✅ FAVORABLE: ResNav significantly {'better' if favorable else 'worse'}")
        print(f"     Improvement: {abs(diff_pct):.1f}%")
    elif favorable:
        print(f"  ⚠️  ResNav better but not statistically significant")
    else:
        print(f"  ❌ APF performs better on this metric")

    return {
        'metric': metric_name,
        'rl_mean': rl_mean,
        'rl_std': rl_std,
        'apf_mean': apf_mean,
        'apf_std': apf_std,
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': d,
        'effect_size': interpretation,
        'favorable': favorable and p_value < 0.05
    }

def analyze_success_rate(rl_successes, rl_total, apf_successes, apf_total, difficulty):
    """Analyze success rate using proportion z-test."""
    print(f"\n{'='*60}")
    print(f"SUCCESS RATE - {difficulty.upper()}")
    print(f"{'='*60}")

    rl_rate = rl_successes / rl_total
    apf_rate = apf_successes / apf_total

    print(f"\nSuccess Rates:")
    print(f"  ResNav: {rl_rate*100:.1f}% ({rl_successes}/{rl_total})")
    print(f"  APF:    {apf_rate*100:.1f}% ({apf_successes}/{apf_total})")

    # Two-proportion z-test
    count = np.array([rl_successes, apf_successes])
    nobs = np.array([rl_total, apf_total])

    # Using chi-square test for 2x2 contingency table
    contingency = np.array([
        [rl_successes, rl_total - rl_successes],
        [apf_successes, apf_total - apf_successes]
    ])

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    print(f"\nChi-Square Test:")
    print(f"  χ² = {chi2:.4f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.001:
        sig_level = "*** (p < 0.001)"
    elif p_value < 0.01:
        sig_level = "** (p < 0.01)"
    elif p_value < 0.05:
        sig_level = "* (p < 0.05)"
    else:
        sig_level = "ns (not significant)"
    print(f"  Significance: {sig_level}")

    # Effect size for proportions (Cohen's h)
    h = 2 * (np.arcsin(np.sqrt(rl_rate)) - np.arcsin(np.sqrt(apf_rate)))

    print(f"\nEffect Size (Cohen's h):")
    print(f"  h = {h:.4f}")

    diff = rl_rate - apf_rate
    print(f"\nResult Summary:")
    if diff > 0 and p_value < 0.05:
        print(f"  ✅ FAVORABLE: ResNav significantly better (+{diff*100:.1f} percentage points)")
    elif diff > 0:
        print(f"  ⚠️  ResNav better but not statistically significant")
    else:
        print(f"  ❌ APF performs better")

    return {
        'difficulty': difficulty,
        'rl_rate': rl_rate,
        'apf_rate': apf_rate,
        'chi2': chi2,
        'p_value': p_value,
        'cohens_h': h,
        'favorable': diff > 0 and p_value < 0.05
    }

def main():
    # Load data
    results_path = Path('runs/final_evaluation/results/detailed_results.csv')

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        return

    df = pd.read_csv(results_path)

    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS OF EVALUATION RESULTS")
    print("="*60)
    print(f"\nTotal samples: {len(df)}")
    print(f"Methods: {df['method'].unique()}")
    print(f"Difficulties: {df['difficulty'].unique()}")

    all_results = []

    # Analyze success rates by difficulty
    print("\n\n" + "#"*60)
    print("# PART 1: SUCCESS RATE ANALYSIS")
    print("#"*60)

    for difficulty in ['basic', 'medium', 'hard']:
        rl_subset = df[(df['method'] == 'rl_agent') & (df['difficulty'] == difficulty)]
        apf_subset = df[(df['method'] == 'apf') & (df['difficulty'] == difficulty)]

        result = analyze_success_rate(
            rl_subset['success'].sum(), len(rl_subset),
            apf_subset['success'].sum(), len(apf_subset),
            difficulty
        )
        all_results.append(result)

    # Analyze continuous metrics (only for successful runs)
    print("\n\n" + "#"*60)
    print("# PART 2: CONTINUOUS METRICS ANALYSIS")
    print("#"*60)
    print("(Computed only for successful episodes)")

    # Get successful runs
    rl_success = df[(df['method'] == 'rl_agent') & (df['success'] == True)]
    apf_success = df[(df['method'] == 'apf') & (df['success'] == True)]

    print(f"\nSuccessful samples: ResNav={len(rl_success)}, APF={len(apf_success)}")

    # Time analysis
    result = analyze_metric(
        rl_success['time_s'],
        apf_success['time_s'],
        'Completion Time (s)',
        higher_is_better=False  # Lower time is better
    )
    all_results.append(result)

    # Path length analysis
    result = analyze_metric(
        rl_success['path_length_m'],
        apf_success['path_length_m'],
        'Path Length (m)',
        higher_is_better=False  # Shorter is better
    )
    all_results.append(result)

    # Smoothness analysis
    result = analyze_metric(
        rl_success['path_smoothness'],
        apf_success['path_smoothness'],
        'Path Smoothness',
        higher_is_better=False  # Lower is better
    )
    all_results.append(result)

    # Summary
    print("\n\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)

    favorable_count = sum(1 for r in all_results if r.get('favorable', False))
    total_tests = len(all_results)

    print(f"\nFavorable results: {favorable_count}/{total_tests}")

    print("\nDetailed Summary:")
    for r in all_results:
        if 'metric' in r:
            status = "✅" if r['favorable'] else "❌"
            print(f"  {status} {r['metric']}: p={r['p_value']:.4f}, d={r['cohens_d']:.2f}")
        elif 'difficulty' in r:
            status = "✅" if r['favorable'] else "❌"
            print(f"  {status} Success Rate ({r['difficulty']}): p={r['p_value']:.4f}, h={r['cohens_h']:.2f}")

    print("\n" + "="*60)
    if favorable_count == total_tests:
        print("🎉 ALL RESULTS ARE STATISTICALLY SIGNIFICANT AND FAVORABLE!")
        print("   Safe to include statistical analysis in report.")
    elif favorable_count > total_tests / 2:
        print("✅ Most results are favorable. Consider including in report.")
    else:
        print("⚠️  Mixed results. Review before including in report.")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
