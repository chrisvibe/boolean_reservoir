import cProfile
import pstats
import io
import torch

'''
Conclusion: on cpu and gpu matmul + float32 should be best even if not mathematically sound :( the optimizations out of box dont focus on binary stuff
'''
class BinaryTester:
    """Binary to integer conversion with multiple strategies."""
    
    def __init__(self, bits: int, input_dtype: torch.dtype | None = None, helper_dtype: torch.dtype | None = None):
        self.input_dtype = input_dtype
        self.helper_dtype = helper_dtype
        self.powers_of_2 = 2 ** torch.arange(bits, dtype=helper_dtype).flip(0)
        self.shifts = torch.arange(bits - 1, -1, -1, dtype=helper_dtype)
    
    def bin2int(self, x: torch.Tensor) -> torch.Tensor:
        """Sum-based reduction."""
        return (x * self.powers_of_2).sum(dim=-1)
    
    def bin2int_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication approach."""
        return torch.matmul(x, self.powers_of_2)
    
    def bin2int_matrix_plus_cast(self, x): # x is not same type as powers_of_2
        return torch.matmul(x.to(self.powers_of_2.dtype), self.powers_of_2).to(torch.int64)
    
    def bin2int_packed(self, x: torch.Tensor) -> torch.Tensor:
        """Bit-packing approach (optimized for CPU)."""
        return (x << self.shifts).sum(dim=-1)
    
    def bin2int_chunked(self, x):
        chunks = torch.split(x, 8, dim=-1)
        packed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_shifts = torch.arange(chunk.shape[-1]-1, -1, -1)
            packed_chunk = (chunk << chunk_shifts).sum(dim=-1) << (8 * i)
            packed_chunks.append(packed_chunk)
        return sum(packed_chunks)

def profile_method(tester: BinaryTester, method_name: str, bits: int, 
                   iterations: int = 100, samples: int = 100, nodes: int = 5000):
    """Profile a conversion method and print results."""
    method = getattr(tester, method_name)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(iterations):
        x = torch.randint(0, 2, (samples, nodes, bits), dtype=tester.input_dtype or torch.uint8)
        _ = method(x)
    
    profiler.disable()
    
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats()
    
    print("\n".join(stream.getvalue().splitlines()[:20]))


def main():
    bits = 31
    methods = [
        # ("bin2int", torch.uint8, None),
        # ("bin2int_matrix", torch.int64, torch.int64),
        # ("bin2int_matrix", torch.float64, torch.float64),
        ("bin2int_matrix", torch.float32, torch.float32),
        # ("bin2int_matrix", torch.float16, torch.float16),
        # ("bin2int_matrix", torch.int8, torch.int8),
        ("bin2int_matrix_plus_cast", torch.uint8, torch.float32),
        # ("bin2int_packed", torch.uint8, None),
        # ("bin2int_chunked", torch.int64, None),
    ]
    
    for method_name, input_dtype, helper_dtype in methods:
        print(f"\n{'='*60}")
        print(f"Testing: {method_name} (input={input_dtype}, helper={helper_dtype})")
        print('='*60)
        
        tester = BinaryTester(bits=bits, input_dtype=input_dtype, helper_dtype=helper_dtype)
        profile_method(tester, method_name, bits)


if __name__ == "__main__":
    main()