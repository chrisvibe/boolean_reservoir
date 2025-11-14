import cProfile
import pstats
import io
from projects.boolean_reservoir.code.parameters import * 
from projects.boolean_reservoir.code.train_model import train_single_model, grid_search, EuclideanDistanceAccuracy as accuracy
from projects.path_integration.code.dataset_init import PathIntegrationDatasetInit as dataset_init

def profile_training_function():
    pr = cProfile.Profile()
    pr.enable()
    p, model, dataset = train_single_model('projects/path_integration/test/config/2D/test_model.yaml', dataset_init=dataset_init, accuracy=accuracy)
    pr.disable()

    # Create IO stream for profiler results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()

    # Print the profiler results
    print(s.getvalue())

if __name__ == '__main__':
    profile_training_function()