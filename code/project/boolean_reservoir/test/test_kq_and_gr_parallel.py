
from project.boolean_reservoir.code.kq_and_gr_metric_parallel import boolean_reservoir_kq_gr_grid_search
from project.boolean_reservoir.code.kq_and_gr_metric import get_kernel_quality_dataset, get_generalization_rank_dataset, DatasetInitKQGR
from project.boolean_reservoir.code.utils.explore_grid_search_data import fix_factors_and_combo
from project.boolean_reservoir.code.parameter import load_yaml_config, save_yaml_config
from project.boolean_reservoir.code.utils.utils import custom_load_grid_search_data
from project.temporal.code.visualization import plot_kq_and_gr, plot_kq_and_gr_many_config

import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(process)d - %(filename)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    dataset_init = DatasetInitKQGR(get_gr_fn=get_kernel_quality_dataset, get_kq_fn=get_generalization_rank_dataset)

    configs = list()
    configs.append('config/temporal/kq_and_gr/grid_search/test.yaml')

    for config in configs:
        print(config)
        # from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
        # param_combinations = generate_param_combinations(load_yaml_config(config))
        # P = boolean_reservoir_kq_gr_grid_search(config, dataset_init, param_combinations=[param_combinations[0]])
        P = boolean_reservoir_kq_gr_grid_search(config, dataset_init)

        P = load_yaml_config(config)
        extractions = [
            ('P', lambda p: p, None),
            ('I', lambda p: p.M.I, {'pertubation', 'encoding', 'redundancy', 'chunks', 'interleaving'}),
            ('R', lambda p: p.M.R, {'mode', 'k_avg', 'init', 'k_max', 'self_loops'}),
            ('M', lambda p: p.L.M, {'kq', 'gr', 'delta', 'spectral_radius'}),
        ]
        df, factors = custom_load_grid_search_data(config_paths=config, extractions=extractions)
        df, factors = fix_factors_and_combo(df, factors=factors, exclude={'L_out_path', 'R_k_avg', 'R_k_max', 'M_kq', 'M_gr', 'M_delta', 'M_spectral_radius'})
        grouped_df = df.groupby(df['combo'])
        plot_kq_and_gr_many_config(grouped_df, P, 'many_config.svg')
        file_path = P.L.out_path / 'visualizations' / 'config'
        file_path.mkdir(parents=True, exist_ok=True)
        for name, subset in grouped_df:
            p = subset.iloc[0]['P']
            i = subset.iloc[0]['combo_id']
            save_yaml_config(p, file_path / f'{i}.yaml')
            plot_kq_and_gr(subset, p, f'config_{i}_kq_and_gr.svg')

