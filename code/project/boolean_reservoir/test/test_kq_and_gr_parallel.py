
from project.boolean_reservoir.code.kq_and_gr_metric_parallel import boolean_reservoir_kq_gr_grid_search
from project.boolean_reservoir.code.kq_and_gr_metric import get_kernel_quality_dataset, get_generalization_rank_dataset, DatasetInitKQGR
from project.boolean_reservoir.code.utils.explore_grid_search_data import load_custom_data, make_combo_column

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
        P = boolean_reservoir_kq_gr_grid_search(config, dataset_init, cpu_memory_per_job_gb=7)

        from project.temporal.code.visualization import plot_kq_and_gr, group_df_data_by_parameters, plot_kq_and_gr_many_config, plot_optimal_k_avg_vs_configuration 
        from project.boolean_reservoir.code.utils.utils import load_yaml_config
        P = load_yaml_config(config)
        extractions = [
            (lambda p: p.M.I, {'pertubation', 'encoding', 'redundancy', 'chunks', 'interleaving'}),
            (lambda p: p.M.R, {'mode', 'k_avg', 'init', 'k_max', 'self_loops'}),
        ]
        df, factors, groups_dict = load_custom_data(configs, extractions)

        # TODO re-write visualizations.py to accept extractions?
        col, factors_subset = make_combo_column(df, factors)
        grouped_df = df.groupby(col)
        plot_kq_and_gr_many_config(grouped_df, P, 'many_config.svg')

