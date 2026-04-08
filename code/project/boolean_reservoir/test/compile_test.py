import time
import shutil
from project.boolean_reservoir.code.train_model_parallel import boolean_reservoir_grid_search

CONFIG = 'project/boolean_reservoir/test/config/compile_test.yaml'
OUT = 'out/temporal/density/grid_search/design_choices/compile_test'

if __name__ == '__main__':
    results = {}
    for compile_model in [False, True]:
        shutil.rmtree(OUT, ignore_errors=True)
        start = time.time()
        boolean_reservoir_grid_search(CONFIG, gpu_memory_per_job_gb=0.5, compile_model=compile_model)
        results[compile_model] = time.time() - start

    print("\n--- compile benchmark ---")
    for flag, elapsed in results.items():
        print(f"  compile_model={flag}: {elapsed:.1f}s")
    diff = results[True] - results[False]
    print(f"  difference: {diff:+.1f}s ({'compile slower' if diff > 0 else 'compile faster'})")
