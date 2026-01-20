from project.boolean_reservoir.code.utils.explore_grid_search_data import load_custom_data, graph_accuracy_vs_k_avg, create_accuracy_vs_k_avg_dashboard
from project.boolean_reservoir.code.visualization import polar_design_plot
from pathlib import Path

if __name__ == '__main__':
    from project.temporal.code.stat import polar_design_plot
    response = 'accuracy'
    out_path = Path('/out/path_integration/stats/design_evaluation/test_optim')
    path = out_path / ''
    print(path)
    extractions = [
        ('T', lambda p: p.M.T.optim, {'name'}),
        ('params', lambda p: p.M.T.optim.params, {'lr', 'weight_decay'}),
        ('T', lambda p: p.M.T, {'batch_size'}),
        ('R', lambda p: p.M.R, {'mode', 'k_avg', 'init'}),
        ('I', lambda p: p.M.I, {'chunks', 'interleaving'}),
    ]
    paths = list()
    paths.append(f'config/path_integration/2D/grid_search/design_choices_prep/test_optim.yaml')
    df, factors, groups_dict = load_custom_data(paths, extractions, response_variable=response)
    # chucks = 2 is terrible
    # graph_accuracy_vs_k_avg(path, df, factors)
    factors = [f for f in factors if f != 'I_chunks']
    df = df[df['I_chunks'] != 2]
    polar_design_plot(out_path, df, factors, success_thresh=.2, title='test optim')
    graph_accuracy_vs_k_avg(path, df, factors)
    app = create_accuracy_vs_k_avg_dashboard(df, factors)
    app.run_server(debug=True)
