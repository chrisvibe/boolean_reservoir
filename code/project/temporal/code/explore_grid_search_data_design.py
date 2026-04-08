from project.boolean_reservoir.code.utils.explore_grid_search_data import graph_accuracy_vs_k_avg, create_scatter_dashboard
from project.boolean_reservoir.code.visualization import polar_design_plot
from project.boolean_reservoir.code.utils.load_save import custom_load_grid_search_data
from pathlib import Path

if __name__ == '__main__':
    paths = list()
    paths.append('config/temporal/density/grid_search/design_choices/all.yaml')
    paths.append('config/temporal/parity/grid_search/design_choices/all.yaml')

    response = 'accuracy'
    out_path = Path('/out/temporal/stats/design_evaluation/all')
    print(out_path)

    extractions = [
        ('T', lambda p: p.L.T, {'accuracy', 'loss'}),
        ('kqgr', lambda p: p.L.kqgr, {'kq', 'gr', 'delta'}),
        ('L', lambda p: p.L, {'universe', 'out_path'}),
        ('L_out_name', lambda p: Path(p.L.out_path).name if p.L.out_path else None, None),
        ('kqgr', lambda p: p.U.kqgr.D, {'tau', 'evaluation'}),
        ('D', lambda p: p.D, {'task', 'window', 'delay'}),
        ('I', lambda p: p.M.I, {'pertubation', 'encoding', 'redundancy', 'chunks', 'interleaving', 'ticks'}),
        ('R', lambda p: p.M.R, {'mode', 'k_avg', 'init', 'k_max', 'self_loops', 'n_nodes'}),
    ]

    df, factors = custom_load_grid_search_data(config_paths=paths, extractions=extractions)
    print('data loaded...')
    app = create_scatter_dashboard(df, factors)
    app.run(debug=False, dev_tools_hot_reload=False)

    # df_filter_mask = lambda df: (df['T_accuracy'] > 0.6)
    # df, factors = custom_load_grid_search_data(config_paths=paths, extractions=extractions, df_filter_mask=df_filter_mask, limit=100000)
    # print('data loaded...')
    # app = create_scatter_dashboard(df, factors)
    # app.run(debug=False, dev_tools_hot_reload=False)

    # df_filter_mask = lambda df: (df['R_k_avg'].between(1, 5)) & (df['I_pertubation'] != 'xor') & (df['I_chunks'] == 1)
    # df, factors = custom_load_grid_search_data(config_paths=paths, extractions=extractions, df_filter_mask=df_filter_mask)
    # print('data loaded...')
    # app = create_scatter_dashboard(df, factors)
    # factors = [f for f in factors if f not in ['I_interleaving', 'I_encoding', 'R_init', 'R_self_loops']]
    # thresh = 0.2
    # polar_design_plot(out_path, df, factors, success_thresh=thresh, title=f'design_choices_thresh={thresh}')
    # graph_accuracy_vs_k_avg(out_path, df, factors)



    # import cProfile
    # cProfile.run("custom_load_grid_search_data(config_paths=config, extractions=extractions)")
    