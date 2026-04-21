#%% 

import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' # Without this, pyplot crashes the kernal
from matplotlib.ticker import FuncFormatter
from itertools import accumulate
from math import log
from scipy import interpolate
from scipy.stats import ttest_ind
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression
from itertools import accumulate
from statistics import mode
from collections import Counter

from utils import load_dicts, rolling_average

# This file tests the impact of exceptions.



rolling_window = 10000

# Assuming the arguments contains two arg_names: one for exceptions, one for pseudo-exceptions, in that order.
plot_dicts, min_max_dict, complete_order = load_dicts({'titles' : ['name_w_exceptions', 'name_wo_exceptions']})

print('Getting results with exceptions...')
exceptions_dict = plot_dicts[0]
exceptions = [rolling_average(wins, window_size=rolling_window) for wins in exceptions_dict['wins_exception']]  


print('\nGetting results with pseudo-exceptions...')
pseudo_exceptions_dict = plot_dicts[1]
pseudo_exceptions = [rolling_average(wins, window_size=rolling_window) for wins in pseudo_exceptions_dict['wins_exception']]  

print('\nGot results!')

# So, we have two lists. Both lists have a list for each agent, showing rolling-average win-rates with exceptions.

#%%



def plot_these(list_1, list_2):
    fig, ax = plt.subplots(len(list_1), 2, figsize=(20, 4 * len(list_1)))
    fig.suptitle('Exceptions vs Pseudo-Exceptions', fontsize=16)

    # Ensure ax is a 2D array even if len(exceptions) == 1
    if len(exceptions) == 1:
        ax = np.expand_dims(ax, axis=0)

    # Define the plot function
    def plot_rolling_average_wins(axis, data, label, color):
        axis.plot([100 * d for d in data], label=label, color=color)
        axis.set_ylim(0, 100)
        axis.set_xlabel('Epochs')
        axis.set_ylabel('Success Rate')

    # Plot all agent data
    for i in range(len(list_1)):
        plot_rolling_average_wins(ax[i, 0], list_1[i], label='Exceptions', color='blue')
        plot_rolling_average_wins(ax[i, 1], list_2[i], label='Pseudo-Exceptions', color='green')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

plot_these(exceptions, pseudo_exceptions)




#%% 



def _savgol_safe(y, frac=0.03, poly=2):
    n = len(y)
    if n < 7 or frac <= 0: return y
    win = max(7, int(np.floor(n * frac)))
    if win % 2 == 0: win += 1
    if win >= n: win = n-1 if n % 2 == 0 else n
    return savgol_filter(y, window_length=win, polyorder=min(poly, win-1), mode='interp')

def u_shape_score(
    y,
    title,
    burnin_frac=0.1,
    min_pos=(0.20, 0.80),
    smooth_frac=0.03,
    search_radius_frac=0.25,
    k_step_frac=0.01,        # ~1% of n
    band_frac=0.02,          # ignore ±2% n around the valley for peaks
    lambda_anchor=2.0,       # penalty weight for k drifting from valley
    min_depth=0.03,
    min_width=0.06
):
    """
    U-shape score via piecewise isotonic regression,
    with (1) valley-anchored breakpoint penalty and
    (2) robust depth using percentile peaks away from the valley.
    Returns: score, iL, iM (valley), iR  (indices on original series)
    """
    y = np.asarray(y, float)
    n0 = len(y)
    if n0 < 40: 
        return 0.0, None, None, None

    # Burn-in
    b = int(n0 * burnin_frac)
    if 0 < b < n0 - 10:
        y = y[b:]
    n = len(y)
    if n < 40: 
        return 0.0, None, None, None

    # Smooth + robust normalize
    ys = _savgol_safe(y, frac=smooth_frac)
    lo_p, hi_p = np.percentile(ys, [5, 95])
    scale = max(hi_p - lo_p, 1e-8)
    z = np.clip((ys - lo_p) / scale, 0.0, 1.0)

    # Interior valley (on smoothed, normalized series)
    lo = int(n * min_pos[0]); hi = int(n * min_pos[1])
    if hi - lo < 5: 
        return 0.0, None, None, None
    i_min = lo + int(np.argmin(z[lo:hi+1]))  # anchor valley

    # Baseline monotone MSE
    x = np.arange(n, dtype=float)
    inc = IsotonicRegression(increasing=True)
    dec = IsotonicRegression(increasing=False)
    mse_base = min(
        np.mean((z - inc.fit_transform(x, z))**2),
        np.mean((z - dec.fit_transform(x, z))**2)
    )

    # Search k near the valley, but penalize drift from i_min
    rad = max(2, int(search_radius_frac * n))
    lo_k = max(lo, i_min - rad); hi_k = min(hi, i_min + rad)
    step = max(1, int(k_step_frac * n))

    best_obj = np.inf
    best_k, best_mse = None, None
    for k in range(lo_k, hi_k + 1, step):
        if k < 3 or n - k < 3: continue
        yL = dec.fit_transform(x[:k], z[:k])
        yR = inc.fit_transform(x[k:],  z[k:])
        y_hat = np.concatenate([yL, yR])
        mse = np.mean((z - y_hat)**2)
        # anchor penalty (scaled to 0..1)
        drift = abs(k - i_min) / n
        obj = mse + lambda_anchor * (drift**2) * mse_base
        if obj < best_obj:
            best_obj, best_k, best_mse = obj, k, mse

    # Optional tiny local refine around best_k
    if best_k is not None:
        k_lo = max(lo_k, best_k - max(2, step))
        k_hi = min(hi_k, best_k + max(2, step))
        for k in range(k_lo, k_hi + 1):
            if k < 3 or n - k < 3: continue
            yL = dec.fit_transform(x[:k], z[:k])
            yR = inc.fit_transform(x[k:],  z[k:])
            y_hat = np.concatenate([yL, yR])
            mse = np.mean((z - y_hat)**2)
            drift = abs(k - i_min) / n
            obj = mse + lambda_anchor * (drift**2) * mse_base
            if obj < best_obj:
                best_obj, best_k, best_mse = obj, k, mse

    if best_k is None or mse_base < 1e-12:
        return 0.0, None, None, None

    improvement = max(0.0, (mse_base - best_mse) / (mse_base + 1e-12))

    # Robust depth measured around the *valley* i_min
    band = max(3, int(band_frac * n))
    L = z[:max(0, i_min - band)]
    R = z[min(n, i_min + band):]
    if L.size < 3 or R.size < 3:
        return 0.0, None, None, None

    val = z[i_min]
    left_peak  = float(np.percentile(L, 90))
    right_peak = float(np.percentile(R, 90))
    depth = max(0.0, min(left_peak - val, right_peak - val))

    width = min(i_min, n - i_min) / n          # width around the actual valley
    width = min(1.0, width / 0.25)

    if depth < min_depth or width < min_width or not (lo <= i_min <= hi):
        return 0.0, None, None, None

    score = float(np.clip(0.6 * improvement + 0.25 * depth + 0.15 * width, 0.0, 1.0))

    # Indices for plotting (choose true local peaks near those percentiles)
    # Left peak:
    iL_region = z[:max(1, i_min - band)]
    iL = int(np.argmax(iL_region)) if iL_region.size else 0
    # Right peak:
    iR_region = z[min(n, i_min + band):]
    iR = int(min(n-1, i_min + band + (np.argmax(iR_region) if iR_region.size else 0)))

    # shift back by burn-in
    iL += b; iM = i_min + b; iR += b
    return score, iL, iM, iR



def compare_u_scores_ttest(group_a, group_b):
    t_stat, p_val_two_tailed = ttest_ind(group_a, group_b, equal_var=False)
    if t_stat > 0:
        p_val_one_tailed = p_val_two_tailed / 2
    else:
        p_val_one_tailed = 1 - p_val_two_tailed / 2  # not significant in this direction

    return {
        'p_value': p_val_one_tailed,
        't_statistic': t_stat,
        'mean_a': np.mean(group_a),
        'mean_b': np.mean(group_b),
        'effect_size': np.mean(group_a) - np.mean(group_b)
    }
    
    

def plot_groups_with_u_indices(
    curves_a, curves_b,
    scores_a, iLs_a, iMs_a, iRs_a,
    scores_b, iLs_b, iMs_b, iRs_b,
    label_a='Exceptions', label_b='Pseudo-Exceptions'
):
    assert len(curves_a) == len(curves_b), 'Groups must have same number of curves'
    n = len(curves_a)

    fig, ax = plt.subplots(n, 2, figsize=(20, 4 * n), sharex=False)
    if n == 1:
        ax = np.expand_dims(ax, axis=0)

    # Set global font size
    plt.rcParams.update({'font.size': 14})

    def _maybe_percent(y):
        y = np.asarray(y, dtype=float)
        return y if np.nanmax(y) > 2.0 else y * 100.0

    def _valid_triplet(iL, iM, iR, N):
        if iL is None or iM is None or iR is None:
            return False
        try:
            iL, iM, iR = int(iL), int(iM), int(iR)
        except Exception:
            return False
        return 0 <= iL < iM < iR < N

    def _plot_one(axis, y, score, iL, iM, iR, color, title):
        yp = _maybe_percent(y)
        axis.plot(yp, color=color, lw=1.6)
        axis.set_ylim(0, 100)
        axis.set_xlabel('Epochs', fontsize=22)
        axis.set_ylabel('Success Rate', fontsize=22)
        axis.set_title(title, fontsize=22)

        # Annotate if indices look valid
        if _valid_triplet(iL, iM, iR, len(yp)):
            iL, iM, iR = int(iL), int(iM), int(iR)
            axis.axvline(iL, color='red', ls='--', lw=3)
            axis.axvline(iM, color='red', ls='--', lw=3)
            axis.axvline(iR, color='red', ls='--', lw=3)
            axis.plot([iL, iM, iR], [yp[iL], yp[iM], yp[iR]], 'o', ms=5, color='red')

        # Larger tick labels
        axis.tick_params(axis='both', which='major', labelsize=22)

    for i in range(n):
        _plot_one(
            ax[i, 0], curves_a[i], scores_a[i], iLs_a[i], iMs_a[i], iRs_a[i],
            color='blue', title=label_a
        )
        _plot_one(
            ax[i, 1], curves_b[i], scores_b[i], iLs_b[i], iMs_b[i], iRs_b[i],
            color='green', title=label_b
        )

    fig.suptitle('Exceptions vs Pseudo-Exceptions (U-shape annotated)', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    
    
exception_u_scores = []
exception_iLs = []
exception_iMs = []
exception_iRs = []
for i, e in enumerate(exceptions):
    score, iL, iM, iR = u_shape_score(e, title = f'exceptions {i}')
    exception_u_scores.append(score)
    exception_iLs.append(iL)
    exception_iMs.append(iM)
    exception_iRs.append(iR)
    
    
    
pseudo_exception_u_scores = []
pseudo_exception_iLs = []
pseudo_exception_iMs = []
pseudo_exception_iRs = []
for i, e in enumerate(pseudo_exceptions):
    score, iL, iM, iR = u_shape_score(e, title = f'pseudo {i}')
    pseudo_exception_u_scores.append(score)
    pseudo_exception_iLs.append(iL)
    pseudo_exception_iMs.append(iM)
    pseudo_exception_iRs.append(iR)



plot_groups_with_u_indices(
    exceptions, pseudo_exceptions,
    exception_u_scores, exception_iLs, exception_iMs, exception_iRs,
    pseudo_exception_u_scores, pseudo_exception_iLs, pseudo_exception_iMs, pseudo_exception_iRs,
    label_a='Exceptions', label_b='Pseudo-Exceptions'
)

print(exception_u_scores)
print(pseudo_exception_u_scores)

results = compare_u_scores_ttest(exception_u_scores, pseudo_exception_u_scores)

print('Permutation Test Results:')
for key, val in results.items():
    print(f'{key}: {val:.4f}')
# %%
