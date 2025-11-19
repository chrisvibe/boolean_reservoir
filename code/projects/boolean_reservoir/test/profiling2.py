import cProfile
import pstats
import io
from pathlib import Path
from projects.boolean_reservoir.code.parameters import * 
from projects.boolean_reservoir.code.train_model import train_single_model, EuclideanDistanceAccuracy as a
from projects.path_integration.code.dataset_init import PathIntegrationDatasetInit as d
from projects.boolean_reservoir.code.reservoir import BooleanReservoir

def profile_training_function(config, out_dir=None):
    # Ensure output directory exists
    methods = {
        'bin2int': lambda x: x.bin2int,
        'bin2int_packed': lambda x: x.bin2int_packed,
        'bin2int_matrix': lambda x: x.bin2int_matrix,
               }
    for alias in methods:
        model = BooleanReservoir(load_path=config)
        model.bin2int = methods[alias](model)
        pr = cProfile.Profile()
        pr.enable()
        p, model, dataset, history = train_single_model(model=model, dataset_init=d().dataset_init, accuracy=a().accuracy)
        pr.disable()

        # Create IO stream for profiler results
        sub_dir = Path(out_dir) if out_dir else model.L.save_path
        sub_dir = sub_dir / 'profiler' / alias
        sub_dir.mkdir(parents=True, exist_ok=True)
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
        
        # Print the profiler results
        first_lines = "\n".join(s.getvalue().splitlines()[:20])
        print(first_lines)
        
        # Save results to file
        with open(sub_dir / 'profile_results.txt', 'w') as f:
            f.write(s.getvalue())
        
        # Also save the binary profile for further analysis
        pr.dump_stats(str(sub_dir / 'profile_results.prof'))
        
        print(f"\nProfile results saved to {sub_dir}/")

if __name__ == '__main__':
    config = 'projects/path_integration/test/config/2D/single_run/test_model_profiling.yaml'
    profile_training_function(config, out_dir='/out')