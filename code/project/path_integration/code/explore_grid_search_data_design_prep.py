from project.boolean_reservoir.code.utils.explore_grid_search_data import graph_accuracy_vs_k_avg, create_accuracy_vs_k_avg_dashboard
from project.boolean_reservoir.code.visualization import polar_design_plot
from project.boolean_reservoir.code.utils.utils import custom_load_grid_search_data
from pathlib import Path

if __name__ == '__main__':
    # config = Path('config/path_integration/2D/grid_search/design_choices_prep/all.yaml')
    config = Path('config/path_integration/2D/grid_search/design_choices_prep/all2.yaml')
    paths = list()
    paths.append(config)
    response = 'accuracy'
    out_path = Path('/out/path_integration/stats/design_evaluation_prep/test_all')
    out_path = Path('/out/path_integration/stats/design_evaluation_prep/test_all2')
    path = out_path / ''
    print(path)

    extractions = [
        ('T', lambda p: p.L.T, {'accuracy', 'loss'}),
        ('I', lambda p: p.M.I, {'pertubation', 'encoding', 'redundancy', 'chunks', 'interleaving'}),
        ('R', lambda p: p.M.R, {'mode', 'k_avg', 'init', 'k_max', 'self_loops'}),
    ]

    df, factors = custom_load_grid_search_data(config_paths=config, extractions=extractions, df_filter_mask=lambda df: df.index < 100)
    print('data loaded...')
    app = create_accuracy_vs_k_avg_dashboard(df, factors)
    app.run(debug=False, dev_tools_hot_reload=False)

    # df_filter = lambda df: (df['R_k_avg'].between(1, 5)) & (df['I_pertubation'] != 'xor') & (df['I_chunks'] == 1)
    # df, factors = custom_load_grid_search_data(config_paths=config, extractions=extractions, df_filter=df_filter)
    # print('data loaded...')
    # app = create_accuracy_vs_k_avg_dashboard(df, factors)
    # factors = [f for f in factors if f not in ['I_interleaving', 'I_encoding', 'R_init', 'R_self_loops']]
    # thresh = 0.2
    # polar_design_plot(out_path, df, factors, success_thresh=thresh, title=f'design_choices_thresh={thresh}')
    # # graph_accuracy_vs_k_avg(path, df, factors)