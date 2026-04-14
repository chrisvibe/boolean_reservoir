from project.boolean_reservoir.code.utils.explore_grid_search_data import create_scatter_dashboard
from project.boolean_reservoir.code.utils.load_save import custom_load_grid_search_data
from pathlib import Path

paths = [
    'config/temporal/density/grid_search/design_choices/all.yaml',
    'config/temporal/parity/grid_search/design_choices/all.yaml',
]

extractions = [
    ('T', lambda p: p.L.T, {'accuracy', 'loss'}),
    ('kqgr', lambda p: p.L.kqgr, {'kq', 'gr', 'delta', 'spectral_radius'}),
    ('L', lambda p: p.L, {'universe', 'out_path'}),
    ('L_out_name', lambda p: Path(p.L.out_path).name if p.L.out_path else None, None),
    ('kqgr', lambda p: p.U.kqgr.D, {'tau', 'evaluation'}),
    ('D', lambda p: p.D, {'task', 'window', 'delay'}),
    ('I', lambda p: p.M.I, {'pertubation', 'encoding', 'redundancy', 'chunks', 'interleaving', 'ticks'}),
    ('R', lambda p: p.M.R, {'mode', 'k_avg', 'init', 'k_max', 'self_loops', 'n_nodes'}),
]

df, factors = custom_load_grid_search_data(config_paths=paths, extractions=extractions)
app = create_scatter_dashboard(df, factors)
server = app.server
