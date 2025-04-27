from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import List, Union, Optional
from pathlib import Path

class Split(BaseModel):
    train: float = Field(0.8, description="data set fraction for training")
    dev: float = Field(0.1, description="data set fraction for development")
    test: float = Field(0.1, description="data set fraction for testing")

    @model_validator(mode='after')
    def must_sum_to_one(cls, values):
        total = float(sum((values.train, values.dev, values.test)))
        if total != 1.0:
            raise ValueError('The sum of train, dev, and test must be 1. Got {}'.format(total))
        return values

class DatasetParameters(BaseModel):
    path: Optional[Path] = Field(None, description="Path to dataset")
    split: Split = Field(Split(), description="fraction for train, dev, test")
    generate_data: bool = Field(False, description="Ignores loading even if dataset exists at path")
    samples: int = Field(64, description="Number of samples to generate in the dataset")
    seed: int = Field(0, description="Random seed, None disables seed")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow'
    )