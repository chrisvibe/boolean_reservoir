from project.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter
from project.boolean_reservoir.code.utils.utils import override_symlink
from pathlib import Path
import torch

def compute_rank(model: BooleanReservoir, x: torch.Tensor, metric: str) -> int:
    """Run model and compute rank from reservoir states"""
    nested_out = model.L.save_path / 'history' / metric
    new_save_path = nested_out / 'history'
    
    record = model.record
    try: # probs overkill, but want to avoid being stuck in recording when doing grid search...
        model.record = True
        if model.history:
            model.history = BatchedTensorHistoryWriter(
                save_path=new_save_path,
                persist_to_disk=model.history.persist_to_disk,
                buffer_size=model.history.buffer_size
            )
        else:
            model.history = BatchedTensorHistoryWriter(save_path=new_save_path, persist_to_disk=False)
        model.eval()
        with torch.no_grad():
            _ = model(x)
        model.flush_history()
    finally:
        model.record = record
    
    override_symlink(Path('../../checkpoint'), new_save_path / 'checkpoint')
    load_dict, history, expanded_meta, meta = model.history.reload_history()
    df_filter = expanded_meta[expanded_meta['phase'] == 'output_layer']
    filtered_history = history[df_filter.index].to(torch.float)
    reservoir_node_history = filtered_history[:, ~model.input_nodes_mask]
    return torch.linalg.matrix_rank(reservoir_node_history).item()