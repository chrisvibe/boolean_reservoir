from pathlib import Path
from projects.boolean_reservoir.code.parameters import load_yaml_config, save_yaml_config
from projects.boolean_reservoir.code.utils import override_symlink 
import pandas as pd
from scipy.stats import f_oneway, levene, shapiro, kruskal, anderson
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import matplotlib
import re
matplotlib.use('Agg')

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def load_data(variable):
    kq_and_gr_paths = list()
    # kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/homogeneous_deterministic.yaml')
    # kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/homogeneous_stochastic.yaml')
    # kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/heterogeneous_deterministic.yaml')
    # kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/heterogeneous_stochastic.yaml')

    kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/homogeneous_deterministic.yaml')
    # kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/homogeneous_stochastic.yaml')
    # kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/heterogeneous_deterministic.yaml')
    # kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/heterogeneous_stochastic.yaml')

    training_paths = list()
    # training_paths.append('config/temporal/density/grid_search/homogeneous_deterministic.yaml')
    # training_paths.append('config/temporal/density/grid_search/homogeneous_stochastic.yaml')
    # training_paths.append('config/temporal/density/grid_search/heterogeneous_deterministic.yaml')
    # training_paths.append('config/temporal/density/grid_search/heterogeneous_stochastic.yaml')

    training_paths.append('config/temporal/parity/grid_search/homogeneous_deterministic.yaml')
    # training_paths.append('config/temporal/parity/grid_search/homogeneous_stochastic.yaml')
    # training_paths.append('config/temporal/parity/grid_search/heterogeneous_deterministic.yaml')
    # training_paths.append('config/temporal/parity/grid_search/heterogeneous_stochastic.yaml')

    d_set = {}
    d_set = {'task', 'tao', 'window_size'}
    i_set = {'connection', 'pertubation'}
    r_set = {'mode', 'k_avg', 'init'}
    r_set = {'k_avg', 'init'}
    factors = [f'D_{x}' for x in d_set] + [f'I_{x}' for x in i_set] + [f'R_{x}' for x in r_set]
    data = list()
    for path in kq_and_gr_paths: # concat data
        _, df_i = load_grid_search_data_from_yaml(path, data_filename='df.h5')
        df_i = process_grid_search_data_kq_and_gr(df_i, d_set, i_set, r_set)
        data.append(df_i)
    df_metric = pd.concat(data, ignore_index=True)

    data = list()
    for path in training_paths: # concat data
        _, df_i = load_grid_search_data_from_yaml(path, data_filename='log.h5')
        df_i = process_grid_search_data(df_i, d_set, i_set, r_set)
        data.append(df_i)
    df_train = pd.concat(data, ignore_index=True)

    # Note: make sure all factors represent main variations s.t. we get normal distributions within the groups
    df_metric = df_metric[df_metric['D_tao'] == 5] # metric for prediction
    df_train = df_train[df_train['D_tao'] == 5] # delay in temporal dataset task
    factors = list(df_train[factors].nunique()[df_train[factors].nunique() > 1].index)

    df = aggregate_and_merge_data(df_metric, df_train, factors)
    groups_dict = {k: v[variable].values for k, v in df.groupby('combo')}
    return df, factors, groups_dict

def process_grid_search_data(df, d_set, i_set, r_set):
    flatten_params = lambda x: pd.concat([
        pd.Series({f"D_{k}": v for k, v in x.D.model_dump().items() if k in d_set}),
        pd.Series({f"I_{k}": v for k, v in x.M.I.model_dump().items() if k in i_set}),
        pd.Series({f"R_{k}": v for k, v in x.M.R.model_dump().items() if k in r_set}),
    ])
    df_flattened_params = df['params'].apply(lambda p: flatten_params(p))
    df = pd.concat([df, df_flattened_params], axis=1)

    df['grid_search'] = df['params'].apply(lambda p: p.L.out_path.name)
    if df.iloc[0]['params'].L.train_log.loss:
        df['loss'] = df['params'].apply(lambda p: p.L.train_log.loss) # BCE
        df['accuracy'] = df['params'].apply(lambda p: p.L.train_log.accuracy)
    df.drop(['params'], axis=1, inplace=True)
    return df

def process_grid_search_data_kq_and_gr(df, d_set, i_set, r_set): # TODO come back to this and compare KQ and GR to training?
    # row matrix style on KQ/GR/Delta so we make new columns to unpack this
    df_pivoted = df.pivot_table(index=['config', 'sample'], columns='metric', values='value').reset_index()
    df = pd.merge(df[['config', 'sample', 'params', 'spectral_radius']].groupby(['config', 'sample']).first(), 
        df_pivoted, 
        on=['config', 'sample'], 
        how='left')
    df = process_grid_search_data(df, d_set, i_set, r_set)
    return df

def load_grid_search_data_from_yaml(path, data_filename='df.h5'):
    P = load_yaml_config(path)
    data_file_path = P.L.out_path / data_filename
    df = pd.read_hdf(data_file_path, 'df')
    return P, df

def aggregate_and_merge_data(df1, df2, factors):
    # max_values = df1.groupby(['D_tao', 'sample'])['delta'].idxmax() # max delta per tao-sample (over many k_avg)
    # max_subset = df1.loc[max_values]
    # max_subset['k_avg*'] = max_subset['k_avg'].mean() # average delta* over the grid_search samples

    agg_dict = {k: 'first' for k in factors}
    agg_dict = agg_dict | {'delta': 'mean', 'gr': 'mean', 'kq': 'mean'}
    df1 = df1.groupby('config').agg(agg_dict)

    df1['combo'] = df1.apply(lambda row: tuple(row[feature] for feature in factors), axis=1)
    df2['combo'] = df2.apply(lambda row: tuple(row[feature] for feature in factors), axis=1)
    df = pd.merge(df1, df2, on=['combo'], how='inner')
    df.columns = [col[:-2] if col.endswith('_x') else col for col in df.columns]
    df.drop([col for col in df.columns if col.endswith('_y')], axis=1, inplace=True)

    # df['group_id'], _ = pd.factorize(df['combo'])
    # df.drop(['combo'], axis=1, inplace=True)
    return df

def anova_analysis_discrete(out_path: Path):
    '''
    factorial ANOVA (interaction effects are interesting)
    ----------------------
    response parameter: accuracy (right-skewed and have heteroscedasticity - increasing spread with magnitude)
       fyi: loss (BCE)
    factors: categorical design choices
    groups: parameter per design combination
    samples per group: 25 independent samples
    avoid normality and homogenity criteria by using generealized anova for discrete data with binomial model

    - KQ/GR/Delta -> accuracy
    What is the relationship between KQ/GR/Delta and Accuracy
    What is the relationship between design choice and Accuracy
    '''

    out_path.mkdir(exist_ok=True)
    response = 'accuracy'
    df, factors, groups_dict = load_data(response)
    df = df[['combo', response, 'delta', 'kq', 'gr'] + factors]

    # TODO write a note here about how combinations + k_avg are too many and we split analysis in two stages

    # group by k_avg to see effect of just that, use best k_avg as base (highest accuracy mean)
    df2 = set_reference_category(df, 'R_k_avg', 2.0) 
    factors = ['R_k_avg']
    df2[factors] = df2[factors].apply(pd.to_numeric)
    # formula = "proportion ~ " + " * ".join(factors) # continuous factors
    formula = "proportion ~ bs(R_k_avg, df=3)"
    model = check_factorial_glm_binomial_fixed_sample_size_per_row(df2, response, formula)
    print("\nANOVA Model:")
    print(model.summary())

    # Calculate predictions for different k values
    k_values = np.arange(1, 11)
    pred_df = pd.DataFrame({'R_k_avg': k_values})
    probabilities = model.predict(pred_df)
    plt.plot(k_values, probabilities)
    plt.savefig(out_path / 'best_k_avg.png')

    # TODO using optimal k_avg find the best design combinations 
    # df2 = set_reference_category(df, 'R_k_avg', 2.0) 
    # formula = "proportion ~ " + " * ".join([f"C({factor})" for factor in factors]) # categorical factors
    # model = check_factorial_glm_binomial_fixed_sample_size_per_row(df2, response, formula)

def anova_analysis_continuous(out_path: Path):
    '''
    factorial ANOVA (interaction effects are interesting)
    ----------------------
    response parameter: loss/accuracy (right-skewed and have heteroscedasticity - increasing spread with magnitude)
       loss (BCE)
       accuracy -> log(p / (1 - p))  # maybe not good if many have max accuracy
    factors: categorical design choices
    groups: parameter per design combination
    samples per group: 25 independent samples
    homogeneity: Levines test (variance similar per spread)
    normality: individual QQ plots + Shapiro Wilk (per Group) + normal distribution of all residuals (model prediction - real)

    - KQ/GR/Delta -> Loss/accuracy
    What is the relationship between KQ/GR/Delta and Loss/Accuracy
    What is the relationship between design choice and Loss/Accuracy

    - What if homogenity or normality is not achieved?
    - plot ols model coefficients with confidence levels
    - plot real vs predicted values from model
    - plot residuals (model predicted - real) to verify normal distribution
    '''

    parameter = 'loss'
    df, factors, groups_dict = load_data(parameter)
    df = df[['combo', parameter, 'delta', 'kq', 'gr'] + factors]

    ######## NORMALITY #############################################
    thresh = 0.02
    normality_results = check_normality(out_path, groups_dict)
    print("\nNormality Test (Shapiro-Wilk h-statistic and p-values by combo):")
    print("\nH0: The data is normally distributed. If p > thresh → H0 holds")
    for name, (h_stat, p_value) in normality_results.items():
        color = GREEN if p_value >= thresh else RED
        print(f"{'-'.join(map(str, name))}: h-statistic={h_stat:.3f}, p-value={color}{p_value:.3f}{RESET}")


    ######## HOMOGENEITY ###########################################
    thresh = 0.02
    homogeneity_h_stat, homogeneity_p_value = check_homogeneity(groups_dict)
    print(f"\nHomogeneity of Variances (Levene's Test):")
    print("H0: All groups have equal variances (homoscedasticity). If p > {thresh} → H0 holds\n")

    color = GREEN if homogeneity_p_value >= thresh else RED
    print(f"h-statistic={homogeneity_h_stat:.3f}, p-value={color}{homogeneity_p_value:.3f}{RESET}")

    ######## H0 ####################################################
    # h_stat, p_value = check_h0(groups_dict)
    # print("\nTest of H0 (kruskal):")
    # print("H0: The distributions of the groups are the same.")
    # print(f"H-statistic={h_stat:.3f}, p-value={p_value:.3f}")

    anova_table, model = check_factorial_anova(df, parameter, factors)
    print("\nTest of H0 (factor ANOVA):")
    print("H0_i: The average values of the dependent variable are the same across all levels of factor i.")
    print("H0_ij: There is no interaction between factor i and factor j, meaning the effect of factor i on the dependent variable is consistent across all levels of factor j, and vice versa.")
    print("\nANOVA Table:")
    print(anova_table)
    print("\nANOVA Model:")
    print(model.summary())

    plot_coefficients(out_path, model)
    plot_predictions(out_path, model, df, parameter)
    plot_residuals(out_path, model, df, parameter)

# Check normality for each combination
# The null hypothesis (H₀) in the context of the Shapiro-Wilk test is that the data come from a normal distribution.
# The test statistic (h_stat) indicates how close the sample distribution is to normality, but you make decisions based on the p-value.
# We reject H₀ if p < 0.05, indicating a significant deviation from normality.
def check_normality(path, groups_dict):
    normality_results = {}
    folder_path = path / 'visualizations/qq-plots/'
    folder_path.mkdir(parents=True, exist_ok=True)
    for name, group in groups_dict.items():
        h_stat, p_value = shapiro(group)
        normality_results[name] = (h_stat, p_value)
        # Q-Q plot for visual assessment
        sm.qqplot(group, line='s')
        plt.title(f'Q-Q plot for {name}')
        plt.subplots_adjust(bottom=0.15)
        plt.figtext(0.5, 0.03, f'Shapiro Test: h={h_stat:.3f}, p={p_value:.3f}', ha='center', fontsize=10)
        plt.savefig(folder_path / f'{sanitize_filename(name)}.png')
        plt.close()
    return normality_results

def sanitize_filename(name):
    # If it's a tuple or list, join with underscores
    if isinstance(name, (tuple, list)):
        name = "_".join(str(part) for part in name)
    # Replace any invalid characters
    return re.sub(r'[<>:"/\\|?*]', '_', str(name))

# Conduct Levene's test for homogeneity of variances across combinations
def check_homogeneity(groups_dict):
    h_stat, p_value = levene(*groups_dict.values())
    return h_stat, p_value

# Perform Kruskal-Wallis test
def check_h0(groups_dict):
    h_stat, p_value = kruskal(*groups_dict.values())
    return h_stat, p_value

def check_factorial_anova(df, metric, factors):
    """Perform factorial ANOVA for multiple factors (all assumed categorical)."""
    # Construct the formula: metric ~ factor1 * factor2 * ... * factorN
    formula = f"{metric} ~ " + " * ".join([f"C({factor})" for factor in factors])
    # Fit the model using OLS
    model = ols(formula, data=df).fit()
    # Perform ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table, model

def check_factorial_glm_binomial_fixed_sample_size_per_row(df, accuracy_col, formula):
    """
    Perform factorial analysis using a Binomial GLM from accuracy values with a fixed number of trials per row.

    Assumptions:
    - 'accuracy' is the proportion of correct predictions per row, computed as correct / total.
    - The number of trials (total) is the same for every row and is not provided as a column.
    - All factors specified are treated as categorical variables.

    This approach models the log-odds of accuracy using a binomial likelihood and supports factorial designs.
    """

    df = df.copy()
    df["proportion"] = df[accuracy_col] # correct/total for m samples (batch size)


    model = smf.glm(
        formula,
        data=df,
        family=sm.families.Binomial()
    )
    model_fit = model.fit()

    return model_fit

def set_reference_category(df, factor_name, reference_level):
    """
    Reorder factor levels to set a specific level as the reference category for GLM
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    factor_name : str
        Name of the factor column to reorder
    reference_level : 
        The level to use as reference (will be first in category order)
    """
    # Get current levels
    current_levels = sorted(df[factor_name].unique())
    
    # Check if reference level exists
    if reference_level not in current_levels:
        raise ValueError(f"Reference level '{reference_level}' not found in {factor_name}. "
                       f"Available levels: {current_levels}")
    
    # Create new order with reference_level first
    new_order = [reference_level] + [level for level in current_levels if level != reference_level]
    
    # Convert to categorical with the new order
    df_copy = df.copy()
    df_copy[factor_name] = pd.Categorical(df_copy[factor_name], 
                                          categories=new_order, 
                                          ordered=False)
    
    print(f"Set '{reference_level}' as reference for {factor_name}")
    
    return df_copy

def plot_predictions(path, model, df, metric):
    """Plot real vs. predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df[metric], model.fittedvalues, alpha=0.3)
    plt.plot(df[metric], df[metric], color='red', linestyle='--', linewidth=1)
    plt.title('Observed vs Predicted Values')
    plt.xlabel('Observed Values')
    plt.ylabel('Predicted Values')
    folder_path = path / 'visualizations'
    folder_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder_path / 'real_vs_predicted_model.png')
    plt.close()

def plot_coefficients(path, model):
    """Plot the coefficients and their confidence intervals."""
    plt.figure(figsize=(15, 15))
    params = model.params
    conf = model.conf_int()
    errors = (conf.iloc[:, 1] - conf.iloc[:, 0]) / 2
    
    x_positions = range(len(params))
    plt.errorbar(x_positions, params, yerr=errors, fmt='o')
    plt.axhline(0, color='grey', linestyle='--')
    plt.title('Coefficients and 95% Confidence Intervals')
    plt.xlabel('Predictors')
    plt.ylabel('Coefficient Estimate')
    plt.xticks([])
    for i, (label, val) in enumerate(zip(params.index, params)):
        plt.text(i, val + errors.iloc[i] + 0.02*max(params), 
                 label, 
                 rotation=90, 
                 ha='center', 
                 va='bottom',
                 fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    folder_path = path / 'visualizations'
    folder_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder_path / 'predictions_and_confidence_levels.png')
    plt.close()

def plot_residuals(path, model, df, metric):
    """Plot residuals and assess normality."""
    residuals = df[metric] - model.fittedvalues
    result = anderson(residuals)
    test_statistic = result.statistic
    
    # Accessing the critical value for 5% significance level
    significance_levels = result.significance_level
    critical_value_05 = result.critical_values[significance_levels.tolist().index(5)]

    plt.figure(figsize=(8, 6))
    sm.qqplot(residuals, line='s')
    plt.title('Q-Q Plot of Residuals')
    plt.subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.03, f'Anderson-Darling Test: statistic={test_statistic:.3f}, critical value(5%)={critical_value_05:.3f}', ha='center', fontsize=10)
    folder_path = path / 'visualizations'
    folder_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder_path / 'qq_plot_residuals.png')
    plt.close()

if __name__ == '__main__':
    out_path = Path('/out/temporal/stats')
    anova_analysis_discrete(out_path)