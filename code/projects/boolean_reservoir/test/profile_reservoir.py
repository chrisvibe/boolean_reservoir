import cProfile
import pstats
import io
from pathlib import Path
from projects.boolean_reservoir.code.parameters import * 
from projects.boolean_reservoir.code.train_model import train_single_model, EuclideanDistanceAccuracy as a
from projects.path_integration.code.dataset_init import PathIntegrationDatasetInit as d

def profile_training_function(out_dir=None):
    pr = cProfile.Profile()
    pr.enable()
    config = 'projects/path_integration/test/config/2D/single_run/test_model_profiling.yaml'
    p, model, dataset, history = train_single_model(config, dataset_init=d().dataset_init, accuracy=a().accuracy)
    pr.disable()
    
    # Ensure output directory exists
    out_dir = Path(out_dir) / 'profiler' if out_dir else model.L.save_path / 'profiler'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create IO stream for profiler results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()
    
    # Print the profiler results
    first_lines = "\n".join(s.getvalue().splitlines()[:20])
    print(first_lines)
    
    # Save results to file
    with open(out_dir / 'profile_results.txt', 'w') as f:
        f.write(s.getvalue())
    
    # Also save the binary profile for further analysis
    pr.dump_stats(str(out_dir / 'profile_results.prof'))
    
    print(f"\nProfile results saved to {out_dir}/")

if __name__ == '__main__':
    profile_training_function()