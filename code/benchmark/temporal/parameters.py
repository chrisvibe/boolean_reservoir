from benchmark.utils.parameters import DatasetParameters
from pydantic import Field, model_validator
from typing import List, Union
from pathlib import Path

class TemporalDatasetParams(DatasetParameters):
    task: str = Field(..., description="two options: density or parity")
    bit_stream_length: Union[int, List[int]] = Field(5, description="Length of the bit stream")
    window_size: Union[int, List[int]] = Field(5, description="Size of the window for temporal data")
    tao: Union[int, List[int]] = Field(0, description="Tao parameter")

    @model_validator(mode='after')
    def update_path_after_init(self):
        self.path = self._generate_path()
        return self

    def _generate_path(self):
        if self.has_list_in_a_field(): # not yet expanded (doesnt check recursively)
            return

        return (Path('data/temporal')
            / self.task
            / f'u-{self.bit_stream_length}'
            / f'w-{self.window_size}'
            / f't-{self.tao}'
            / f'm-{self.samples}'
            / f'r-{self.seed}'
            / 'dataset.pt'
        )

