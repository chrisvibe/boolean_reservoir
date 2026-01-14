from project.boolean_reservoir.code.utils.explore_grid_search_data import load_custom_data, graph_accuracy_vs_k_avg, create_accuracy_vs_k_avg_dashboard
from project.boolean_reservoir.code.visualization import polar_design_plot
from pathlib import Path

if __name__ == '__main__':
    config = Path('config/path_integration/2D/grid_search/design_choices_prep/all.yaml')
    config = Path('config/path_integration/2D/grid_search/design_choices_prep/all2.yaml')
    paths = list()
    paths.append(config)
    response = 'accuracy'
    out_path = Path('/out/path_integration/stats/design_evaluation_prep/test_all')
    out_path = Path('/out/path_integration/stats/design_evaluation_prep/test_all2')
    path = out_path / ''
    print(path)

    extractions = [
        (lambda p: p.M.I, {'pertubation', 'encoding', 'redundancy', 'chunks', 'interleaving'}),
        (lambda p: p.M.R, {'mode', 'k_avg', 'init', 'k_max', 'self_loops'}),
    ]

    df, factors, groups_dict = load_custom_data(paths, extractions, response_variable=response)
    print('data loaded...')
    app = create_accuracy_vs_k_avg_dashboard(df, factors)
    app.run(debug=False, dev_tools_hot_reload=False)

    # df_filter = lambda df: (df['R_k_avg'].between(1, 5)) & (df['I_pertubation'] != 'xor') & (df['I_chunks'] == 1)
    # df, factors, groups_dict = load_custom_data(paths, extractions, df_filter=df_filter, response_variable=response)
    # print('data loaded...')
    # app = create_accuracy_vs_k_avg_dashboard(df, factors)
    # factors = [f for f in factors if f not in ['I_interleaving', 'I_encoding', 'R_init', 'R_self_loops']]
    # thresh = 0.2
    # polar_design_plot(out_path, df, factors, success_thresh=thresh, title=f'design_choices_thresh={thresh}')
    # # graph_accuracy_vs_k_avg(path, df, factors)