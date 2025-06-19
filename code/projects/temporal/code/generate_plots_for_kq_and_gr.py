from projects.boolean_reservoir.code.parameters import load_yaml_config, save_yaml_config
from projects.temporal.code.visualizations import plot_kq_and_gr, group_df_data_by_parameters, plot_kq_and_gr_many_config, plot_optimal_k_avg_vs_configuration 
from projects.temporal.code.kq_and_gr_metrics import override_samples_in_p 
from pathlib import Path
import pandas as pd

def load_data_from_yaml(path):
    P = load_yaml_config(path)
    P = override_samples_in_p(P)
    data_file_path = P.L.out_path / 'df.h5'
    df = pd.read_hdf(data_file_path, 'df')
    df.loc[:, 'k_avg'] = df['params'].apply(lambda p: p.M.R.k_avg)
    df.loc[:, 'tao'] = df['params'].apply(lambda p: p.D.tao)
    return P, df
    
if __name__ == '__main__':
    paths = list()
    # paths.append('config/temporal/kq_and_gr/test.yaml')

    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/heterogeneous_stochastic.yaml')

    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/heterogeneous_stochastic.yaml')

    paths.append('config/temporal/kq_and_gr/vary_tao/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/vary_tao/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/vary_tao/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/vary_tao/heterogeneous_stochastic.yaml')

    # for path in paths:
        # P, df = load_data_from_yaml(path)
        # grouped_df = group_df_data_by_parameters(df)
        # plot_kq_and_gr_many_config(grouped_df, P, 'many_config.png')
        # grouped_subset = group_df_data_by_parameters(df[df['metric'] == 'kq'])
        # plot_kq_and_gr_many_config(grouped_subset, P, 'many_config_kq_only.png')
        # grouped_subset = group_df_data_by_parameters(df[df['metric'] == 'gr'])
        # plot_kq_and_gr_many_config(grouped_subset, P, 'many_config_gr_only.png')
        # grouped_subset = group_df_data_by_parameters(df[df['metric'] == 'delta'])
        # plot_kq_and_gr_many_config(grouped_subset, P, 'many_config_delta_only.png')

        # file_path = P.L.out_path / 'config'
        # file_path.mkdir(exist_ok=True, parents=True)
        # for i, (name, subset) in enumerate(grouped_df):
        #     p = subset.iloc[0]['params']
        #     save_yaml_config(p, file_path / f'{i}.yaml')
        #     plot_kq_and_gr(subset, P, f'config_{i}_kq_and_gr.png')
        
        # subset = df[df['params'].apply(lambda p: p.M.R.mode == 'homogeneous')]
        # plot_kq_and_gr(subset, P, 'homogeneous_kq_and_gr.png')
        # subset = df[df['params'].apply(lambda p: p.M.R.mode == 'heterogeneous')]
        # plot_kq_and_gr(subset, P, 'heterogeneous_kq_and_gr.png')

        # subset = df[df['params'].apply(lambda p: p.M.R.mode == 'homogeneous') & (df['metric'] == 'kq')]
        # plot_kq_and_gr(subset, P, 'homogeneous_kq.png')
        # subset = df[df['params'].apply(lambda p: p.M.R.mode == 'heterogeneous') & (df['metric'] == 'kq')]
        # plot_kq_and_gr(subset, P, 'heterogeneous_kq.png')

        # subset = df[df['params'].apply(lambda p: p.M.R.mode == 'homogeneous') & (df['metric'] == 'gr')]
        # plot_kq_and_gr(subset, P, 'homogeneous_gr.png')
        # subset = df[df['params'].apply(lambda p: p.M.R.mode == 'heterogeneous') & (df['metric'] == 'gr')]
        # plot_kq_and_gr(subset, P, 'heterogeneous_gr.png')

    # # smooth curve for example config
    # paths = list()
    # paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/homogeneous_deterministic.yaml')
    # paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/homogeneous_deterministic.yaml')
    # for path in paths:
    #     P, df = load_data_from_yaml(path)
    #     grouped_df = group_df_data_by_parameters(df)
    #     for config_idx in [1, 9]: # not config in data as k_avg bumbs the config number (config in grouping)
    #         df_i = list(grouped_df)[config_idx][1]
    #         grouped_df_i = group_df_data_by_parameters(df_i) 
    #         plot_kq_and_gr_many_config(grouped_df_i, P, f'smoothed_config_{config_idx}.png')

    # special
    ##########################
    paths = list()
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/heterogeneous_stochastic.yaml')

    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/heterogeneous_stochastic.yaml')

    data = list()
    for path in paths: # concat data
        _, df = load_data_from_yaml(path)
        data.append(df)
    big_df = pd.concat(data, ignore_index=True)
    path = Path('out/temporal/kq_and_gr/grid_search')
    plot_optimal_k_avg_vs_configuration(path, big_df)

        
        