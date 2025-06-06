"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
# Used ChatGPT to help guide where to put the adaptations to the given sdt_ddm.py file 


# Mapping dictionaries for categorical variables
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    data = pd.read_csv(file_path)
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    if prepare_for == 'sdt':
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        return pd.DataFrame(sdt_data)
    elif prepare_for == 'delta plots':
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', *[f'p{p}' for p in PERCENTILES]])
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                overall_rt = c_data['rt']
                accurate_rt = c_data[c_data['accuracy'] == 1]['rt']
                error_rt = c_data[c_data['accuracy'] == 0]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['overall'],
                    **{f'p{p}': [np.percentile(overall_rt, p)] for p in PERCENTILES}
                })])
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['accurate'],
                    **{f'p{p}': [np.percentile(accurate_rt, p)] for p in PERCENTILES}
                })])
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['error'],
                    **{f'p{p}': [np.percentile(error_rt, p)] for p in PERCENTILES}
                })])
        return dp_data

def apply_hierarchical_sdt_model(data):
    # Get unique participants and conditions
    P = len(data['pnum'].unique()) 
    C = len(data['condition'].unique())  

    # Define design matrix for fixed effects (Stimulus Type, Difficulty, Interaction)
    stimulus_type = np.array([0, 1, 0, 1])  # 0=simple, 1=complex
    difficulty = np.array([0, 0, 1, 1])     # 0=easy, 1=hard
    interaction = stimulus_type * difficulty

# Quanitfying the effect of Stimulus Type and Trial Difficulty: 
    with pm.Model() as sdt_model:
        # ==== Fixed effects for d′ ====
        beta_d0 = pm.Normal('beta_d0', 0, 1)
        beta_d_stim = pm.Normal('beta_d_stim', 0, 1)
        beta_d_diff = pm.Normal('beta_d_diff', 0, 1)
        beta_d_inter = pm.Normal('beta_d_inter', 0, 1)

        mean_d_prime = beta_d0 + beta_d_stim * stimulus_type + beta_d_diff * difficulty + beta_d_inter * interaction
        stdev_d_prime = pm.HalfNormal('stdev_d_prime', sigma=1.0)

        # ==== Fixed effects for criterion (c) ====
        beta_c0 = pm.Normal('beta_c0', 0, 1)
        beta_c_stim = pm.Normal('beta_c_stim', 0, 1)
        beta_c_diff = pm.Normal('beta_c_diff', 0, 1)
        beta_c_inter = pm.Normal('beta_c_inter', 0, 1)

        mean_criterion = beta_c0 + beta_c_stim * stimulus_type + beta_c_diff * difficulty + beta_c_inter * interaction
        stdev_criterion = pm.HalfNormal('stdev_criterion', sigma=1.0)

        # ==== Participant-level parameters ====
        d_prime = pm.Normal('d_prime', mu=mean_d_prime, sigma=stdev_d_prime, shape=(P, C))
        criterion = pm.Normal('criterion', mu=mean_criterion, sigma=stdev_criterion, shape=(P, C))

        # ==== Likelihoods ====
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)

        pm.Binomial('hit_obs',
                    n=data['nSignal'],
                    p=hit_rate[data['pnum'] - 1, data['condition']],
                    observed=data['hits'])

        pm.Binomial('false_alarm_obs',
                    n=data['nNoise'],
                    p=false_alarm_rate[data['pnum'] - 1, data['condition']],
                    observed=data['false_alarms'])

    return sdt_model

def analyze_and_visualize(file_path):
    # === Read and prepare data for SDT ===
    sdt_data = read_data(file_path, prepare_for='sdt')

    # === Fit the hierarchical SDT model ===
    with apply_hierarchical_sdt_model(sdt_data) as model:
        trace = pm.sample(2000, tune=1000, target_accept=0.9, return_inferencedata=True)

    # === Plot trace diagnostics ===
    trace_vars = [
        "beta_d0", "beta_d_stim", "beta_d_diff", "beta_d_inter",
        "beta_c0", "beta_c_stim", "beta_c_diff", "beta_c_inter"
    ]
    az.plot_trace(trace, var_names=trace_vars)
    plt.tight_layout()
    plt.savefig("traceplots.png")
    plt.close()

# Checking convergence of SDT model: 
    # === Posterior summary table ===
    summary = az.summary(trace, var_names=trace_vars, hdi_prob=0.95)
    print("\nPosterior Summary of Fixed Effects (Regression Coefficients):")
    print(summary)
    summary.to_csv("trace_summary.csv")

    # === Read and prepare data for delta plots ===
    dp_data = read_data(file_path, prepare_for='delta plots')

    # === Draw delta plots per participant ===
    for p in dp_data['pnum'].unique():
        draw_delta_plots(dp_data, pnum=p)
    
    return trace, dp_data



def draw_delta_plots(data, pnum):
    data = data[data['pnum'] == pnum]
    conditions = data['condition'].unique()
    n_conditions = len(conditions)
    fig, axes = plt.subplots(n_conditions, n_conditions, figsize=(4*n_conditions, 4*n_conditions))
    OUTPUT_DIR = Path(__file__).parent.parent.parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    marker_style = {'marker': 'o','markersize': 10,'markerfacecolor': 'white','markeredgewidth': 2,'linewidth': 3}
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if j == 0:
                axes[i,j].set_ylabel('Difference in RT (s)', fontsize=12)
            if i == len(axes)-1:
                axes[i,j].set_xlabel('Percentile', fontsize=12)
            if i > j:
                continue
            if i == j:
                axes[i,j].axis('off')
                continue
            cmask1 = data['condition'] == cond1
            cmask2 = data['condition'] == cond2
            overall_mask = data['mode'] == 'overall'
            error_mask = data['mode'] == 'error'
            accurate_mask = data['mode'] == 'accurate'
            overall_delta = np.array([data[cmask2 & overall_mask][f'p{p}'].values[0] - data[cmask1 & overall_mask][f'p{p}'].values[0] for p in PERCENTILES])
            error_delta = np.array([data[cmask2 & error_mask][f'p{p}'].values[0] - data[cmask1 & error_mask][f'p{p}'].values[0] for p in PERCENTILES])
            accurate_delta = np.array([data[cmask2 & accurate_mask][f'p{p}'].values[0] - data[cmask1 & accurate_mask][f'p{p}'].values[0] for p in PERCENTILES])
            axes[i,j].plot(PERCENTILES, overall_delta, color='black', **marker_style)
            axes[j,i].plot(PERCENTILES, error_delta, color='red', **marker_style)
            axes[j,i].plot(PERCENTILES, accurate_delta, color='green', **marker_style)
            axes[j,i].legend(['Error', 'Accurate'], loc='upper left')
            axes[i,j].set_ylim(bottom=-1/3, top=1/2)
            axes[j,i].set_ylim(bottom=-1/3, top=1/2)
            axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[j,i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[i,j].text(50, -0.27, f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', ha='center', va='top', fontsize=12)
            axes[j,i].text(50, -0.27, f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', ha='center', va='top', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'delta_plots_{pnum}.png')


# Comparing the effects of Trial Difficulty manipulation with Stimulus Type manipulation: 
def compare_sdt_and_delta(trace, dp_data):
    """
    Compare the effect of stimulus type and difficulty using SDT and RT (delta plots).
    """
    print("\n===== SDT Model Parameter Effects (Posterior Summaries) =====")
    sdt_summary = az.summary(trace, var_names=[
        "beta_d_stim", "beta_d_diff", "beta_d_inter",
        "beta_c_stim", "beta_c_diff", "beta_c_inter"
    ])
    print(sdt_summary)


    print("\n===== Delta Plot Mean RT Differences by Condition =====")
    condition_pairs = [
        (0, 1),  # Easy Simple vs Easy Complex (stimulus type effect)
        (0, 2),  # Easy Simple vs Hard Simple (difficulty effect)
    ]

    for cond1, cond2 in condition_pairs:
        label = f"{CONDITION_NAMES[cond2]} - {CONDITION_NAMES[cond1]}"
        deltas = []
        for pnum in dp_data['pnum'].unique():
            pdata = dp_data[dp_data['pnum'] == pnum]
            mask1 = (pdata['condition'] == cond1) & (pdata['mode'] == 'overall')
            mask2 = (pdata['condition'] == cond2) & (pdata['mode'] == 'overall')
            if mask1.any() and mask2.any():
                rt1 = np.array([pdata[mask1][f'p{p}'].values[0] for p in PERCENTILES])
                rt2 = np.array([pdata[mask2][f'p{p}'].values[0] for p in PERCENTILES])
                deltas.append(rt2 - rt1)
        if deltas:
            avg_delta = np.mean(deltas, axis=0)
            print(f"{label} RT Differences at Percentiles (10-90): {np.round(avg_delta, 3)}")

if __name__ == "__main__":
    trace, dp_data = analyze_and_visualize("/home/jovyan/cogs107s25/sdt_ddm_project/data/data.csv")
    compare_sdt_and_delta(trace, dp_data)

    print("\n==== Interpretation ====")
    print("Trial difficulty has a greater effect on the SDT parameters (especially d') and RTs than the stimulus type.")
    print("Stimulus type has a smaller, but more stable, effect. The higher the difficulty, the weaker the stimulus effects are.")

