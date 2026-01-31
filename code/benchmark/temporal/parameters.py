from benchmark.utils.parameters import DatasetParameters
from pydantic import Field, model_validator
from typing import List, Union
from pathlib import Path

class TemporalDatasetParams(DatasetParameters):
    dimensions: Union[int, List[int]] = Field(1, description="Number of independent bit streams")
    task: str = Field(..., description="two options: density or parity")
    bits: Union[int, List[int]] = Field(5, description="length of the bit stream")
    window: Union[int, List[int]] = Field(5, description="size of the window for temporal data")
    delay: Union[int, List[int]] = Field(0, description="delay; shifts window from right to left; higher delay is easier as bits are processed right to left")
    sampling_mode: Union[str, List[str]] = Field( 'random',
    description="'random': random bit patterns with repetition allowed. "
                "'exhaustive': enumerate patterns 0 to 2^bits-1, taking first 'samples' patterns (or cycling if samples > 2^bits)"
    )

    @model_validator(mode='after')
    def update_path_after_init(self):
        self.path = self._generate_path()
        return self

    def _generate_path(self):
        if self.has_list_in_a_field(): # not yet expanded (doesnt check recursively)
            return

        return (Path('data/temporal')
            / self.task
            / f'{self.dimensions}D'
            / f's-{self.sampling_mode}'
            / f'b-{self.bits}'
            / f'w-{self.window}'
            / f'd-{self.delay}'
            / f'm-{self.samples}'
            / f'r-{self.seed}'
            / 'dataset.pt'
        )
    
    def update_path(self):
        self.path = self._generate_path()

