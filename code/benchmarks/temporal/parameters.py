from pydantic import BaseModel, Field, model_validator
from typing import List, Union, Optional
from pathlib import Path, PosixPath, WindowsPath
import yaml

def represent_pathlib_path(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

yaml.add_representer(Path, represent_pathlib_path)
yaml.add_representer(PosixPath, represent_pathlib_path)
yaml.add_representer(WindowsPath, represent_pathlib_path)

class TemporalDatasetParams(BaseModel):
    task: str = Field(..., description="two options: density or parity")
    bit_stream_length: int = Field(5, description="Length of the bit stream")
    window_size: int = Field(5, description="Size of the window for temporal data")
    tao: int = Field(0, description="Tao parameter")
    samples: int = Field(100, description="Number of samples in the dataset")
    seed: int = Field(0, description="Random seed, None disables seed")
    generate_data: bool = Field(False, description="Ignores loading even if dataset exists at path")
    dataset: Path = Field(None, description="Path to dataset")

    @model_validator(mode='before')
    def handle_alias(cls, values):
        if 'u' in values:
            values['bit_stream_length'] = values.pop('u')
        if 'w' in values:
            values['window_size'] = values.pop('w')
        if 't' in values:
            values['tao'] = values.pop('t')
        if 'm' in values:
            values['samples'] = values.pop('m')
        if 'r' in values:
            values['seed'] = values.pop('r')
        return values

    @model_validator(mode='after')
    def handle_dataset_is_none(cls, values):
        if values.dataset is None:
            values.dataset = f'data/temporal/{values.task}/u-{values.bit_strean_length}/w-{values.window_size}/t-{values.tao}/m-{values.samples}/r-{values.seed}'
        return values