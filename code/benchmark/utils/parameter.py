from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional
from pathlib import Path
import hashlib
import json


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
    shuffle: bool = Field(True, description="Shuffle dataset before splitting")
    generate_data: bool = Field(False, description="Ignores loading even if dataset exists at path")
    samples: int = Field(64, description="Number of samples to generate in the dataset")
    seed: Optional[int] = Field(None, description="Random seed, None disables seed")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow'
    )

    def has_list_in_a_field(self):
        return any(isinstance(value, list) for value in self.model_dump().values())

    @staticmethod
    def _hash_params(params, n_chars=5):
        """Hash a dictionary for path generation"""
        json_str = json.dumps(params.model_dump(), sort_keys=True)
        hash_obj = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
        return hash_obj[:n_chars]