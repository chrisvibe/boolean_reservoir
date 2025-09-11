from benchmarks.utils.parameters import DatasetParameters
from pydantic import Field, model_validator
from typing import List, Union, Optional
from pathlib import Path

class TemporalDatasetParams(DatasetParameters):
    task: str = Field(..., description="two options: density or parity")
    bit_stream_length: Union[int, List[int]] = Field(5, description="Length of the bit stream")
    window_size: Union[int, List[int]] = Field(5, description="Size of the window for temporal data")
    tao: Union[int, List[int]] = Field(0, description="Tao parameter")

    @model_validator(mode='after')
    def handle_path_is_none(cls, values):
        values.path = Path(f'data/temporal/{values.task}/u-{values.bit_stream_length}/w-{values.window_size}/t-{values.tao}/m-{values.samples}/r-{values.seed}/dataset.pt')
        return values
    
    def update_path(self):
        self.path = None
        self.handle_path_is_none(self)