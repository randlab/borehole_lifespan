# Standard imports
import itertools
import numpy as np
import pandas as pd
import autograd.numpy as np
# Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
# Scipy imports
from scipy.integrate import quad, simpson
from scipy.optimize import minimize
from scipy.stats import norm, lognorm
import scipy.stats as stats 
from scipy.stats import truncnorm
from scipy.integrate import quad
# Autograd import 
from autograd.scipy.stats import norm as autograd_norm
# Lifelines imports
from lifelines import (
    ExponentialFitter, GeneralizedGammaFitter, KaplanMeierFitter,
    LogLogisticFitter, LogNormalFitter, NelsonAalenFitter,
    PiecewiseExponentialFitter, SplineFitter, WeibullFitter
)
from lifelines.fitters import ParametricUnivariateFitter
from lifelines.plotting import rmst_plot, qq_plot
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times, restricted_mean_survival_time
# Initialization of KaplanMeierFitter
kmf = KaplanMeierFitter()

# Simulates decommissioning dates for boreholes with missing values
def simul_decom(data, mask, mu, sigma, max_v):
    new_sampling = []
    
    for index, row in data.loc[mask].iterrows():
        min_v = row['ComDate']
        a, b = (min_v - mu) / sigma, (max_v - mu) / sigma
        tirage = truncnorm.rvs(a, b, loc=mu, scale=sigma)
        rounded_sample = int(round(tirage))
        new_sampling.append(rounded_sample)
    
    data.loc[mask, 'DecomDate'] = new_sampling

# Adjust time according to yield decrease
def adjust_time(row):
    if row['status'] == 0:  #
        dimq = row['DimQ']
        
        if pd.isna(dimq):  
            return row['time']
        
        time = row['time']

        if dimq >= 0 and dimq <= 20:
            # 
            return time
        elif dimq > 20 and dimq <= 50:
            # 
            return time * (1 - dimq / 100)
        else:
            # 
            return time * 0.5
    else:
        # 
        return row['time']


def clean_time_column(data, column_name="time", small_value=0.01):
    """
    Cleans the specified column in the DataFrame by:
    1. Handle negative values, and replace  by 0.
    2. Adding a small positive value to zero elements in the column time.
    3. Removing rows with na values in the column time.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column to clean. Default is "time".
    - small_value (float): The small positive value to add to zero elements. Default is 0.01.
    
    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    """
    # Handle negative values 
    if (data[column_name] < 0).any():
        print(f"Negative values found in the '{column_name}' column. Replacing them with 0.")
        data[column_name] = data[column_name].apply(lambda x: 0 if x < 0 else x)
    else:
        print(f"No negative values found in the '{column_name}' column.")
        
    # Handle zeros
    if (data[column_name] == 0).any():
        print(f"There are zero values in the '{column_name}' column.")
        data[column_name] = np.where(data[column_name] == 0, data[column_name] + small_value, data[column_name])
        print(f"Zero values were replaced by {small_value}.")
    else:
        print(f"There are no zero values in the '{column_name}' column.")
    
    # Handle NA values
    if data[column_name].isna().any():
        number_of_na_values = data[column_name].isna().sum()
        print(f"There are {number_of_na_values} NA values in the '{column_name}' column.")
        data.dropna(subset=[column_name], inplace=True)
        print("NA values were deleted.")
    else:
        print(f"There are no NA values in the '{column_name}' column.")
    return data


#Survival function for the Gompertz distribution
def gompertz_survival_function(t, alpha, lambda_):
    return np.exp(-alpha * (np.exp(lambda_ * t) - 1))


# # Negative log-likelihood for parameter fitting
def neg_log_likelihood_gompertz(params, t, e):
    alpha, lambda_ = params
    if alpha <= 0 or lambda_ <= 0:
        return np.inf
    # Compute the PDF
    pdf_vals = alpha * lambda_ * np.exp(lambda_ * t) * np.exp(-alpha * (np.exp(lambda_ * t) - 1))
    # Compute the survival function
    surv_vals = gompertz_survival_function(t, alpha, lambda_)
    # log-likelihood
    ll = np.sum(e * np.log(pdf_vals + 1e-15)) + np.sum((1 - e) * np.log(surv_vals + 1e-15))
    return -ll

# Plots the survival function of a Kaplan-Meier fit object with customizable options for appearance and output.
def plot_survival_function(kmf_fit, legend_label, color='darkgreen', fig_width_cm=18, fig_height_cm=10,
                                xlim=60, show_censors=True, ci_show=True, output_path=None, legend_loc='lower right',
                                show_gompertz=False, x_grid=None, S_gompertz=None, ci_lower=None, ci_upper=None,
                                gompertz_ci_show=False):
    
    fig_width_in = fig_width_cm / 2.54
    fig_height_in = fig_height_cm / 2.54

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    kmf_fit.plot_survival_function(ax=ax, color=color, label='', show_censors=show_censors,
                                   censor_styles={'ms': 7}, linewidth=1.0, ci_show=ci_show)

    plt.xlabel('Years', fontsize=10, fontname='Times New Roman')
    plt.ylabel('Survival Probability', fontsize=10, fontname='Times New Roman')
    plt.ylim(0, 1.05)
    plt.xlim(0, xlim)
    plt.yticks(fontsize=10, fontname='Times New Roman')
    plt.xticks(np.arange(0, xlim + 1, 10), fontsize=10, fontname='Times New Roman')

    survival_line = mlines.Line2D([], [], color=color, linestyle='-', label=legend_label, linewidth=1.0)
    translucent_rect = mpatches.Patch(color=color, alpha=0.3, label='95% Confidence Interval')

    legend_elements = [survival_line, translucent_rect]

    if show_gompertz and x_grid is not None and S_gompertz is not None:
        ax.plot(x_grid, S_gompertz, label='Gompertz', color='black', linestyle='--', linewidth=1.2)
        legend_elements.append(
            mlines.Line2D([], [], color='black', linestyle='--', label='Gompertz model')
        )

        if gompertz_ci_show and ci_lower is not None and ci_upper is not None:
            ax.fill_between(x_grid, ci_lower, ci_upper, color='black', alpha=0.2, label='Gompertz 95% CI')
            legend_elements.append(
                mpatches.Patch(color='black', alpha=0.2, label='Gompertz 95% CI')
            )

    if show_censors:
        censored_line = mlines.Line2D([], [], color=color, marker='+', linestyle='None', markersize=6,
                                      label='Censored Data')
        legend_elements.append(censored_line)

    legend = plt.legend(handles=legend_elements, loc=legend_loc, fontsize=10,
                        prop={'family': 'Times New Roman'})
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format='tiff', dpi=600)

    plt.show()



# Compute confidence intervals
def confidence_interval_gompertz(t, alpha_hat, lambda_hat, cov_matrix, confidence=0.95):
    """
    Calculates the confidence intervals for the Gompertz survival function.
    """
    z = norm.ppf(1 - (1 - confidence) / 2)  
    
    grad = np.array([
        -(np.exp(lambda_hat * t) - 1) * np.exp(-alpha_hat * (np.exp(lambda_hat * t) - 1)),
        -t * alpha_hat * np.exp(lambda_hat * t) * np.exp(-alpha_hat * (np.exp(lambda_hat * t) - 1))
    ]).T

    var = np.einsum('ij,ij->i', grad @ cov_matrix, grad)
    
    surv = gompertz_survival_function(t, alpha_hat, lambda_hat)
    lower = surv - z * np.sqrt(var)
    upper = surv + z * np.sqrt(var)
    
    return np.clip(lower, 0, 1), np.clip(upper, 0, 1)

# Restricted Mean Survival Time
def gompertz_rmst(t_max, alpha, lambda_):
    t = np.linspace(0, 60, 60)
    surv_func = np.exp(-alpha * (np.exp(lambda_ * t) - 1))
    rmst = np.trapz(surv_func, t)
    return rmst
    

def calcul_rmst_gompertz_ci(T, E, time_limit=60, n_bootstraps=100, confidence_level=0.95):
    """
    Calculates the RMST (Restricted Mean Survival Time) and its confidence interval
for a Gompertz model
    """
    # Fit initial Gompertz
    initial_guess = [1.0, 0.01]
    res = minimize(neg_log_likelihood_gompertz, x0=initial_guess, args=(T, E),
                   bounds=[(1e-5, None), (1e-5, None)])
    
    if not res.success:
        raise RuntimeError("Initial Gompertz optimization failed.")

    alpha_hat, lambda_hat = res.x
    rmst_est = gompertz_rmst(time_limit, alpha_hat, lambda_hat)

    # Bootstrap
    rmst_samples = []
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(T), size=len(T), replace=True)
        T_boot = T[idx]
        E_boot = E[idx]
        try:
            res_b = minimize(neg_log_likelihood_gompertz, x0=initial_guess, args=(T_boot, E_boot),
                             bounds=[(1e-5, None), (1e-5, None)])
            if res_b.success:
                a_b, l_b = res_b.x
                rmst_b = gompertz_rmst(time_limit, a_b, l_b)
                if np.isfinite(rmst_b):
                    rmst_samples.append(rmst_b)
        except:
            continue

    if len(rmst_samples) == 0:
        raise ValueError("No valid bootstrap sample")

    rmst_samples = np.array(rmst_samples)
    se = np.std(rmst_samples)
    
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    lower = rmst_est - z * se
    upper = rmst_est + z * se
    error_margin = (upper - lower) / 2

    return {
        'RMST': rmst_est,
        'SE': se,
        'CI': (lower, upper),
        '±': error_margin
    }

def plot_results_per_group(T1_par_variable, E1_par_variable, tmax=60,
                                                  fig_width_cm=17.2 , fig_height_cm=13.5 ,
                                                  legend_loc='upper right', ax=None, output_path=None,
                                                  n_bootstraps=100, confidence_level=0.95):
    """
    Plots Kaplan-Meier survival curves and Gompertz model fits for each group of the specified variable.
    Also computes RMST from Gompertz fit and its confidence interval.

    Returns:
        dict: contains RMST and confidence interval per group.
    """

    
    colors = ['darkgreen', 'grey', 'darkorange', 'darkred', 'firebrick', 'darkblue']
    kmf = KaplanMeierFitter()


    # Convert the figure dimensions from cm to inches
    fig_width_in = fig_width_cm / 2.54
    fig_height_in = fig_height_cm / 2.54

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    rmst_results = {}

    for i, (group, T1_group) in enumerate(T1_par_variable.items()):
        E1_group = E1_par_variable[group]

        # Fit Kaplan-Meier and plot
        kmf.fit(T1_group, event_observed=E1_group, label=group)
        kmf.plot_survival_function(ax=ax, color=colors[i], show_censors=False, censor_styles={'ms': 5.5})

        # Fit Gompertz model
        initial_guess = [1.0, 0.01]
        res = minimize(neg_log_likelihood_gompertz,
                       x0=initial_guess,
                       args=(T1_group, E1_group),
                       bounds=[(1e-5, None), (1e-5, None)])

        if not res.success:
            print(f"Warning: Gompertz fit failed for group {group}")
            continue

        alpha_hat, lambda_hat = res.x
        x_vals = np.linspace(0, tmax, 500)
        surv_vals = gompertz_survival_function(x_vals, alpha_hat, lambda_hat)
        ax.plot(x_vals, surv_vals, linestyle='--', color=colors[i])

        # RMST point estimate
        rmst_hat, _ = quad(gompertz_survival_function, 0, tmax, args=(alpha_hat, lambda_hat))

        # Bootstrap RMST
        rmst_bootstrap = []
        for _ in range(n_bootstraps):
            idx = np.random.choice(len(T1_group), size=len(T1_group), replace=True)
            T_boot = T1_group.iloc[idx]
            E_boot = E1_group.iloc[idx]

            try:
                res_b = minimize(neg_log_likelihood_gompertz,
                                 x0=initial_guess,
                                 args=(T_boot, E_boot),
                                 bounds=[(1e-5, None), (1e-5, None)])
                if res_b.success:
                    a_b, l_b = res_b.x
                    rmst_b, _ = quad(gompertz_survival_function, 0, tmax, args=(a_b, l_b))
                    if np.isfinite(rmst_b):
                        rmst_bootstrap.append(rmst_b)
            except:
                continue

        rmst_bootstrap = np.array(rmst_bootstrap)
        se = np.std(rmst_bootstrap)
        z = norm.ppf(1 - (1 - confidence_level) / 2)
        ci_lower = rmst_hat - z * se
        ci_upper = rmst_hat + z * se
        ci_pm = (ci_upper - ci_lower) / 2

        rmst_results[group] = {
            'RMST': rmst_hat,
            'SE': se,
            'CI': (ci_lower, ci_upper),
            '±': ci_pm
        }


    ax.set_xlabel('Years', fontsize=9, fontname='Times New Roman')
    ax.set_ylabel('Survival Probability', fontsize=9, fontname='Times New Roman')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, tmax)
    ax.tick_params(axis='both', labelsize=9)
    ax.set_xticks(np.arange(0, tmax + 1, 10))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xticklabels(np.arange(0, tmax + 1, 10), fontname='Times New Roman')
    ax.set_yticklabels(np.round(np.linspace(0, 1, 6), 2), fontname='Times New Roman')

    legend = ax.legend(loc=legend_loc, fontsize=9, prop={'family': 'Times New Roman'})
    legend.get_frame().set_edgecolor('none')
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, format='tiff', dpi=600)
    plt.show()

    print("\nRMST estimates with 95% CI:\n")
    for group, result in rmst_results.items():
        rmst = result['RMST']
        pm = result['±']
        ci = result['CI']
        print(f"{group}: RMST = {rmst:.2f} ± {pm:.2f} ")

    return rmst_results


def TandE_pervariable(data, group_var, time_var='time', status_var='status'):
    """
    Creates T1_group and E1_group variables for each specified group.

    Returns:
        tuple: Two dictionaries containing the T1_group and E1_group variables for each group.
    """
    groups = data.groupby(group_var)  # Group the data by the specified variable

    T_by_group = {}  # Dictionary to store T1 variables
    E_by_group = {}  # Dictionary to store E1 variables

    for group, group_data in groups:  # Loop through each group
        T_by_group[group] = group_data[time_var].reset_index(drop=True)
        E_by_group[group] = group_data[status_var].reset_index(drop=True)

    return T_by_group, E_by_group