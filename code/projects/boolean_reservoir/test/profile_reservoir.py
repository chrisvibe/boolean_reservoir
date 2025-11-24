import cProfile
import pstats
import io
import torch
from pathlib import Path
from projects.boolean_reservoir.code.parameters import * 
from projects.boolean_reservoir.code.train_model import train_single_model, EuclideanDistanceAccuracy as a
from projects.path_integration.code.dataset_init import PathIntegrationDatasetInit as d

def main(config, out_dir=None, gpu=False, label=''):
    pr = cProfile.Profile()
    d().dataset_init(load_yaml_config(config)) # warmup to avoid recording dataset generation
    pr.enable()
    p, model, dataset, history = train_single_model(config, dataset_init=d().dataset_init, accuracy=a().accuracy, ignore_gpu=gpu)
    pr.disable()
    
    # Ensure output directory exists
    out_dir = Path(out_dir) if out_dir else model.L.save_path
    out_dir = out_dir / ('profiler' + ('_' + label) if label else '')
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
    config = 'projects/path_integration/test/config/2D/single_run/test_model_profiling.yaml'
    if torch.cuda.is_available():
        main(config, gpu=True, label='gpu')
    main(config, gpu=False, label='cpu')