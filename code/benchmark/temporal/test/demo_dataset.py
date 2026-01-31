from benchmark.temporal.temporal_density_parity_datasets import (
    TemporalDatasetParams,
    TemporalDensityDataset,
    TemporalParityDataset,
)
from project.boolean_reservoir.code.utils.utils import set_seed
from benchmark.temporal.test.test_temporal_dataset import color_the_stream, format_task_result 


def show_dataset_samples(dataset, num_samples=5):
    """Display samples from a dataset in a readable format."""
    print(f"\n=== {dataset.D.task.upper()} TASK DEMO ===")
    print(f"Dimensions: {dataset.D.dimensions}, Bits: {dataset.D.bits}, "
          f"Window: {dataset.D.window}, Delay: {dataset.D.delay}")
    print(f"Dataset shape - x: {dataset.x.shape}, y: {dataset.y.shape}\n")
    
    num_samples = min(num_samples, len(dataset.x))
    
    for i in range(num_samples):
        sample_x = dataset.x[i, 0]  # d x b
        sample_y = dataset.y[i]      # d
        
        if dataset.D.dimensions == 1:
            bits = sample_x[0].numpy().astype(bool)
            label = sample_y[0].item()
            colored_stream = color_the_stream(bits, dataset.D.window, dataset.D.delay)
            result_text = format_task_result(label, dataset.D.task)
            print(f"Sample {i+1}: {colored_stream} -> {result_text}")
        else:
            print(f"Sample {i+1}:")
            for d in range(dataset.D.dimensions):
                bits = sample_x[d].numpy().astype(bool)
                label = sample_y[d].item()
                colored_stream = color_the_stream(bits, dataset.D.window, dataset.D.delay)
                result_text = format_task_result(label, dataset.D.task)
                print(f"  Stream {d}: {colored_stream} -> {result_text}")
            print()


if __name__ == '__main__':
    # 1D examples
    print("\n" + "="*60)
    print("1D STREAM EXAMPLES")
    print("="*60)
    
    set_seed(0)
    D = TemporalDatasetParams(
        path="/tmp/demo_density_1d",
        task="density",
        bits=10,
        window=5,
        delay=2,
        dimensions=1,
        samples=5,
        sampling_mode='random',
        generate_data=True,
    )
    dataset = TemporalDensityDataset(D)
    show_dataset_samples(dataset)
    
    set_seed(0)
    D = TemporalDatasetParams(
        path="/tmp/demo_parity_1d",
        task="parity",
        bits=10,
        window=5,
        delay=2,
        dimensions=1,
        samples=5,
        sampling_mode='random',
        generate_data=True,
    )
    dataset = TemporalParityDataset(D)
    show_dataset_samples(dataset)
    
    # 2D examples
    print("\n" + "="*60)
    print("MULTI-DIMENSIONAL STREAM EXAMPLES")
    print("="*60)
    
    set_seed(0)
    D = TemporalDatasetParams(
        path="/tmp/demo_density_2d",
        task="density",
        bits=8,
        window=3,
        delay=1,
        dimensions=2,
        samples=3,
        sampling_mode='random',
        generate_data=True,
    )
    dataset = TemporalDensityDataset(D)
    show_dataset_samples(dataset)
    
    set_seed(0)
    D = TemporalDatasetParams(
        path="/tmp/demo_parity_2d",
        task="parity",
        bits=8,
        window=3,
        delay=1,
        dimensions=2,
        samples=3,
        sampling_mode='random',
        generate_data=True,
    )
    dataset = TemporalParityDataset(D)
    show_dataset_samples(dataset)
    
    # Exhaustive mode example
    print("\n" + "="*60)
    print("EXHAUSTIVE SAMPLING MODE")
    print("="*60)
    
    set_seed(0)
    D = TemporalDatasetParams(
        path="/tmp/demo_exhaustive",
        task="density",
        bits=4,
        window=2,
        delay=1,
        dimensions=1,
        samples=16,
        sampling_mode='exhaustive',
        generate_data=True,
    )
    dataset = TemporalDensityDataset(D)
    show_dataset_samples(dataset, num_samples=16)