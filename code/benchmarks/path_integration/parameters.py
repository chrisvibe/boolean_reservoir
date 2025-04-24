from projects.boolean_reservoir.code.dataset_parameters import DatasetParameters
from pydantic import Field, model_validator
from typing import List, Union, Optional
from pathlib import Path

class PathIntegrationDatasetParams(DatasetParameters):
    pass # TODO implement