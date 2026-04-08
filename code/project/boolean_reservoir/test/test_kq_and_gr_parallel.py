from project.boolean_reservoir.code.train_model_parallel import boolean_reservoir_grid_search
from project.boolean_reservoir.code.utils.explore_grid_search_data import fix_factors_and_combo
from project.boolean_reservoir.code.parameter import load_yaml_config, save_yaml_config
from project.boolean_reservoir.code.utils.load_save import custom_load_grid_search_data
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
    config = 'config/temporal/kq_and_gr/grid_search/test.yaml'
    # config = 'config/path_integration/2D/grid_search/design_choices_prep/discrete.yaml'

    P = boolean_reservoir_grid_search(yaml_path=config, cpu_memory_per_job_gb=4)
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
        p = subset.iloc[0]['P'].to_pydantic()
        i = subset.iloc[0]['combo_id']
        save_yaml_config(p, file_path, file_name=f'{i}')
        plot_kq_and_gr(subset, p, f'config_{i}_kq_and_gr.svg')

