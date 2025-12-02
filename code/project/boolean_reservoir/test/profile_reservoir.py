import cProfile
import pstats
import io
import torch
from pathlib import Path
from project.boolean_reservoir.code.parameter import * 
from project.boolean_reservoir.code.train_model import train_single_model, EuclideanDistanceAccuracy as a
from project.path_integration.code.dataset_init import PathIntegrationDatasetInit as d
from project.boolean_reservoir.code.reservoir import BooleanReservoir

# TODO these dont work, torch.compule doesnt always return a model, it may return a forward?
def profile_compile_main(config):
    for compile_model in [True, False]:
        if torch.cuda.is_available():
            profile_compile(config, gpu=True, label=f'gpu_compile-{compile_model}', compile_model=compile_model, iterations=25)
        profile_compile(config, gpu=False, label=f'cpu_compile-{compile_model}', compile_model=compile_model,iterations=5)

def profile_compile(config, out_dir=None, gpu=False, label='', compile_model=True, iterations=1):
    pr = cProfile.Profile()
    ignore_gpu = not gpu
    _, model, dataset, _ = train_single_model(config, dataset_init=d().dataset_init, accuracy=a().accuracy, ignore_gpu=ignore_gpu, compile_model=compile_model, reset_dynamo=compile_model)
    dataset_init = lambda p: dataset
    # pr.enable()
    for _ in range(iterations): # trains on dirty model just to see how quick it is with compiled architecture
        # train_single_model(model=model, dataset_init=dataset_init, accuracy=a().accuracy, ignore_gpu=ignore_gpu, compile_model=False, reset_dynamo=False)
        train_single_model(model=model, dataset_init=dataset_init, accuracy=a().accuracy, ignore_gpu=ignore_gpu, compile_model=True, reset_dynamo=False)
    # pr.disable()
    _save_and_summarize(pr, model, out_dir, label)

def profile_train_single_model_main(config):
    if torch.cuda.is_available():
        profile_train_single_model(config, gpu=True, label='gpu')
    profile_train_single_model(config, gpu=False, label='cpu')

def profile_train_single_model(config, out_dir=None, gpu=False, label=''):
    pr = cProfile.Profile()
    d().dataset_init(load_yaml_config(config)) # warmup to avoid recording dataset generation
    pr.enable()
    p, model, dataset, history = train_single_model(config, dataset_init=d().dataset_init, accuracy=a().accuracy, ignore_gpu=gpu)
    pr.disable()
    _save_and_summarize(pr, model, out_dir, label)
    
def _save_and_summarize(pr, model, out_dir, label):
    # Ensure output directory exists
    out_dir = Path(out_dir) if out_dir else model.P.L.save_path
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
    config = 'project/path_integration/test/config/2D/single_run/test_model_profiling.yaml'
    profile_compile_main(config)
    # profile_train_single_model_main(config)