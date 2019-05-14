import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LogNorm

def log_heatmap(data, log10_vmin=None, log10_vmax=None, num_ticks=5, **kwargs):
    if log10_vmin is None:
        finite_data = data[np.isfinite(data)]
        log10_vmin = max([np.nanmin(np.log10(finite_data[finite_data > 0])), -16])
    if log10_vmax is None:
        finite_data = data[np.isfinite(data)]
        log10_vmax = min([np.nanmax(np.log10(finite_data[finite_data > 0])), 16])
    log10_vmin = int(np.floor(log10_vmin))
    log10_vmax = int(np.ceil(log10_vmax))
    vmin, vmax = 10**log10_vmin, 10**log10_vmax
    cbar_ticks = [10**i for i in range(log10_vmin, log10_vmax+1)]
    return sns.heatmap(data=data, vmin=vmin, vmax=vmax,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cbar_kws={'ticks': cbar_ticks},
            **kwargs)

def log_boxplot(data, max_lag = 20, ax=None):
    max_lag = min([max_lag, min(data.shape[0:2])])
    dfs = []
    for lag in range(0, max_lag):
        if len(data.shape) == 3:
            for batch_id in range(data.shape[2]):
                df = pd.DataFrame()
                df['lognorm'] = np.log10(np.diag(data[:,:, batch_id], -lag))
                df['lag'] = lag
                dfs.append(df)
        else:
            df = pd.DataFrame()
            df['lognorm'] = np.log10(np.diag(data, -lag))
            df['lag'] = lag
            dfs.append(df)
    df = pd.concat(dfs, ignore_index = True)

    if ax is None:
        fig, ax = plt.subplots(1,1)
    sns.boxplot(x = 'lag', y='lognorm', data=df, ax=ax)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_ylabel('log10(norm d loss[i]/d theta[i-lag])')
    return ax

def log_lineplot(data, max_lag = 20, trace=False, ax=None):
    max_lag = min([max_lag, min(data.shape[0:2])])
    dfs = []
    for lag in range(0, max_lag):
        if len(data.shape) == 3:
            for batch_id in range(data.shape[2]):
                df = pd.DataFrame()
                df['lognorm'] = np.log10(np.diag(data[:,:, batch_id], -lag))
                df['lag'] = lag
                df['i'] =  range(batch_id*df.shape[0]+lag,
                        (batch_id+1)*df.shape[0]+lag)
                dfs.append(df)
        else:
            df = pd.DataFrame()
            df['lognorm'] = np.log10(np.diag(data, -lag))
            df['lag'] = lag
            df['i'] = range(lag, df.shape[0]+lag)
            dfs.append(df)
    df = pd.concat(dfs, ignore_index = True)

    if ax is None:
        fig, ax = plt.subplots(1,1)
    if trace:
        sns.lineplot(x = 'lag', y='lognorm', data=df, ax=ax,
            estimator=None, ci=None, units='i', alpha=0.1, color='gray')
    sns.lineplot(x = 'lag', y='lognorm', data=df, ax=ax,
            estimator='median', ci='sd', color='C0')
    ax.axhline(0, color='k', linestyle='--')
    ax.set_ylabel('log10(norm d loss[i]/d theta[i-lag])')
    return ax


def ratios_boxplot(norm_grads, max_lag = 20):
    ratios = norm_grads[:,0:-1]/norm_grads[:,1:]
    max_lag = min([max_lag, min(ratios.shape)])
    dfs = []
    for lag in range(1, max_lag):
        df = pd.DataFrame()
        df['ratio'] = np.diag(ratios, -lag)
        df['lag'] = lag
        dfs.append(df)
    df = pd.concat(dfs, ignore_index = True)

    fig, ax = plt.subplots(1,1)
    sns.boxplot(x='lag', y='ratio', data=df, ax=ax)
    ax.axhline(1, color='k', linestyle='--')
    ax.set_ylim(0,3)
    ax.set_ylabel('norm grad t / norm grad t+1')
    return fig, ax

def ratios_lineplot(norm_grads, max_lag = 20):
    ratios = norm_grads[:,0:-1]/norm_grads[:,1:]
    max_lag = min([max_lag, min(ratios.shape)])
    dfs = []
    for lag in range(1, max_lag):
        df = pd.DataFrame()
        df['ratio'] = np.diag(ratios, -lag)
        df['lag'] = lag
        dfs.append(df)
    df = pd.concat(dfs, ignore_index = True)

    fig, ax = plt.subplots(1,1)
    sns.lineplot(x='lag', y='ratio', data=df, ax=ax,
            ci='sd', estimator='median')
    ax.axhline(1, color='k', linestyle='--')
    ax.set_ylim(0,3)
    ax.set_ylabel('norm grad t / norm grad t+1')
    return fig, ax

def logratios_lineplot(norm_grads, max_lag = 20):
    ratios = np.log10(norm_grads[:,0:-1]/norm_grads[:,1:])
    max_lag = min([max_lag, min(ratios.shape)])
    dfs = []
    for lag in range(1, max_lag):
        df = pd.DataFrame()
        df['ratio'] = np.diag(ratios, -lag)
        df['lag'] = lag
        dfs.append(df)
    df = pd.concat(dfs, ignore_index = True)

    fig, ax = plt.subplots(1,1)
    sns.lineplot(x='lag', y='ratio', data=df, ax=ax,
            ci='sd', estimator='median')
    ax.axhline(0, color='k', linestyle='--')
    ax.set_ylim(-3,2)
    ax.set_ylabel('log10 (norm grad t / norm grad t+1)')
    return fig, ax

def logratios_boxplot(norm_grads, max_lag = 20):
    ratios = np.log10(norm_grads[:,0:-1]/norm_grads[:,1:])
    max_lag = min([max_lag, min(ratios.shape)])
    dfs = []
    for lag in range(1, max_lag):
        df = pd.DataFrame()
        df['ratio'] = np.diag(ratios, -lag)
        df['lag'] = lag
        dfs.append(df)
    df = pd.concat(dfs, ignore_index = True)

    fig, ax = plt.subplots(1,1)
    sns.boxplot(x = 'lag', y='ratio', data=df, ax=ax)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_ylim(-3,2)
    ax.set_ylabel('log10 (norm grad t / norm grad t+1)')
    return fig, ax



