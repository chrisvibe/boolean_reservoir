from pathlib import Path
from parameters import load_yaml_config, Params
import pandas as pd
from scipy.stats import f_oneway, levene, shapiro, kruskal
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols

def init_get_data_and_make_groups(in_path, out_path):
    out_path = Path(out_path)
    out_path = out_path / 'stats' 
    out_path.mkdir(parents=True, exist_ok=True)

    P = load_yaml_config(in_path)
    L = P.logging

    file_path = L.out_path / 'log.h5'
    df = pd.read_hdf(file_path, key='df', mode='r')
    df['loss'] = df['loss'].apply(lambda x: x ** .5) # MSE to RMS
    col = 'loss' # TODO do this for parameter fields?
    df[col] = (df[col]-df[col].mean())/(df[col].std())
    df['model_params'] = df['params'].apply(lambda p_dict: Params(**p_dict).model)
    # input layer
    df['interleaving'] = df['model_params'].apply(lambda p: p.reservoir_layer.k_avg)
    # reservoir layer
    df['n_nodes_eff'] = df['model_params'].apply(lambda p: p.reservoir_layer.n_nodes-p.input_layer.bits_per_feature*p.input_layer.n_inputs) # litterature consideres exclude input nodes in count
    df['k_avg'] = df['model_params'].apply(lambda p: p.reservoir_layer.k_avg)
    df['self_loops'] = df['model_params'].apply(lambda p: p.reservoir_layer.self_loops)
    df['init'] = df['model_params'].apply(lambda p: p.reservoir_layer.init)
    vars = ['loss', 'interleaving', 'k_avg', 'self_loops', 'init']
    df = df[vars]

    df_encoded = pd.get_dummies(df, columns=df.columns[1:])
    df_melted = pd.melt(df_encoded, id_vars='loss', var_name='parameter', value_name='value')
    df_melted = df_melted[df_melted['value'] == True]
    df_melted.drop('value', axis=1, inplace=True)
    df_groups = df_melted.groupby('parameter')['loss'].apply(list)
    return df, df_melted, df_groups, vars

def perform_anova(df, metric):
    """Perform one-way ANOVA on the given metric."""
    model = ols(f'{metric} ~ C(config)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

def anova_analysis(in_path, out_path):
    # compare between loss groups
    ####################################
    # samples: Indepenent
    # variables/factors: Independent & categorical, continuous variables follow normal distribution
    # one-way, two-way, multi-way ANOVA: analysis of variance after changing one or more variables
    # Homogeneity of Variances: The groups should have approximately equal variances → Levene's Test
    # normality: The dependent variable (loss) normally distributed within each group. → Shapiro-Wilk test or by examining Q-Q plots.
    # normality not met: non-parametric tests like Kruskal-Wallis.
    df, df_melted, df_groups, vars = init_get_data_and_make_groups(in_path, out_path)

    # samples sizes per category
    print('\n'.join([str((k, len(df_groups[k]))) for k in df_groups.index]))

    # # H0: is there a difference between the groups (parameter configurations)
    # anova = f_oneway(*df_groups) 
    # print(anova) 

    # how do the groups contribute (how do parameters affect loss)
    # Performing Tukey HSD test
    tukey = pairwise_tukeyhsd(endog=df_melted['loss'], groups=df_melted['parameter'], alpha=0.05)
    print(tukey)

    # model_loss = ols('value ~ config', data=df_groups[df_groups['metric'] == 'loss']).fit()
    # table_loss = sm.stats.anova_lm(model_loss, typ=2)
    # print(table_loss)


if __name__ == '__main__':
    anova_analysis('/out/grid_search/2D/initial_sweep/parameters.yaml', '/tmp')