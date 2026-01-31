from project.boolean_reservoir.code.parameter import load_yaml_config, save_yaml_config
from project.boolean_reservoir.code.utils.utils import load_grid_search_data
from project.temporal.code.visualization import plot_kq_and_gr, group_df_data_by_parameters, plot_kq_and_gr_many_config, plot_optimal_k_avg_vs_configuration 
from project.temporal.code.generate_data_kq_and_gr import override_samples_in_p 
from pathlib import Path
import pandas as pd

def load_data_from_yaml(path):
    P = load_yaml_config(path)
    P = override_samples_in_p(P)
    data_file_path = P.L.out_path / 'log.yaml'
    df, _ = load_grid_search_data(data_paths=data_file_path)
    df.loc[:, 'k_avg'] = df['params'].apply(lambda p: p.M.R.k_avg)
    df.loc[:, 'delay'] = df['params'].apply(lambda p: p.D.delay)
    return P, df
    
if __name__ == '__main__':
    paths = list()
    # paths.append('config/temporal/kq_and_gr/test.yaml')

    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/heterogeneous_stochastic.yaml')

    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/heterogeneous_stochastic.yaml')

    paths.append('config/temporal/kq_and_gr/vary_delay/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/vary_delay/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/vary_delay/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/vary_delay/heterogeneous_stochastic.yaml')


    for path in paths:
        P, df = load_data_from_yaml(path)
        grouped_df = group_df_data_by_parameters(df)
        plot_kq_and_gr_many_config(grouped_df, P, 'many_config.svg')
        grouped_subset = group_df_data_by_parameters(df[df['metric'] == 'kq'])
        plot_kq_and_gr_many_config(grouped_subset, P, 'many_config_kq_only.svg')
        grouped_subset = group_df_data_by_parameters(df[df['metric'] == 'gr'])
        plot_kq_and_gr_many_config(grouped_subset, P, 'many_config_gr_only.svg')
        grouped_subset = group_df_data_by_parameters(df[df['metric'] == 'delta'])
        plot_kq_and_gr_many_config(grouped_subset, P, 'many_config_delta_only.svg')

        file_path = P.L.out_path / 'config'
        file_path.mkdir(exist_ok=True, parents=True)
        for i, (name, subset) in enumerate(grouped_df):
            p = subset.iloc[0]['params']
            save_yaml_config(p, file_path / f'{i}.yaml')
            plot_kq_and_gr(subset, P, f'config_{i}_kq_and_gr.svg')
        
    # smooth curve for example config
    paths = list()
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/homogeneous_deterministic.yaml')
    for path in paths:
        P, df = load_data_from_yaml(path)
        grouped_df = group_df_data_by_parameters(df)
        for config_idx in [1, 9]: # not config in data as k_avg bumbs the config number (config in grouping)
            df_i = list(grouped_df)[config_idx][1]
            grouped_df_i = group_df_data_by_parameters(df_i) 
            plot_kq_and_gr_many_config(grouped_df_i, P, f'smoothed_config_{config_idx}.svg')

    # # special
    # ##########################
    # paths = list()
    # paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/homogeneous_deterministic.yaml')
    # paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/homogeneous_stochastic.yaml')
    # paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/heterogeneous_deterministic.yaml')
    # paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/heterogeneous_stochastic.yaml')

    # paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/homogeneous_deterministic.yaml')
    # paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/homogeneous_stochastic.yaml')
    # paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/heterogeneous_deterministic.yaml')
    # paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/heterogeneous_stochastic.yaml')

    # data = list()
    # for path in paths: # concat data
    #     _, df = load_data_from_yaml(path)
    #     data.append(df)
    # big_df = pd.concat(data, ignore_index=True)
    # path = Path('out/temporal/kq_and_gr/grid_search')
    # plot_optimal_k_avg_vs_configuration(path, big_df)

        
        