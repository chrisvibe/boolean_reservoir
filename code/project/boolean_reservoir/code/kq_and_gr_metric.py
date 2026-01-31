from torch.utils.data import Dataset
from project.temporal.code.dataset_init import TemporalDatasetInit as d
from project.boolean_reservoir.code.parameter import load_yaml_config, Params
from typing import Callable, Tuple

def get_kernel_quality_dataset(p: Params):
    return d().dataset_init(p)

def get_generalization_rank_dataset(p: Params):
    return get_kernel_quality_dataset(p) # handled internally in dataset_init

class DatasetInitKQGR:
    """Dataset initializer that returns both KQ and GR datasets"""

    def __init__(self, get_kq_fn: Callable = get_kernel_quality_dataset, get_gr_fn: Callable = get_generalization_rank_dataset):
        self.get_kq_fn = get_kq_fn
        self.get_gr_fn = get_gr_fn

    def __call__(self, P: Params) -> Tuple[Dataset, Dataset]:
        """Generate/load KQ and GR datasets"""
        kq_dataset = self.get_kq_fn(P)
        gr_dataset = self.get_gr_fn(P)
        return kq_dataset, gr_dataset