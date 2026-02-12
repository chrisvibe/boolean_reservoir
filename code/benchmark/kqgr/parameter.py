from benchmark.utils.parameter import DatasetParameters
from pydantic import Field
from typing import List, Union

class KQGRDatasetParams(DatasetParameters):
    """
    Kernel Quality (KQ) and Generalization Rank (GR) metric dataset parameters.
    
    Hijacks TemporalDensityDataset for data generation but only uses x values (input 
    bitstreams) - y values (labels) are ignored since KQ/GR metrics evaluate reservoir 
    kernel quality and generalization capability, not task performance.
    
    tau/mode control GR metric: during encoding, tau bits per feature are set identical 
    across all samples to test generalization under reduced input diversity.
    """
    dimensions: Union[int, List[int]] = Field(1, description="Number of independent bit streams")
    bits: Union[int, List[int]] = Field(5, description="length of the bit stream")
    sampling_mode: Union[str, List[str]] = Field('random',
        description="'random': random bit patterns with repetition allowed. "
                    "'exhaustive': enumerate patterns 0 to 2^bits-1, taking first 'samples' patterns (or cycling if samples > 2^bits)"
    )
    tau: Union[int, List[int]] = Field(3, description="Number of identical bits for GR metric")
    evaluation: Union[str, List[str]] = Field('last', description="Tau application mode: 'first', 'last', or 'random'")

    @property
    def mode(self):
        return self.evaluation.split('-')[0]