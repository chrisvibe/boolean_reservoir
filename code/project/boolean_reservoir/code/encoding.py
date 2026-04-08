import torch
from project.boolean_reservoir.code.parameter import Params, InputParams
from project.boolean_reservoir.code.utils.primes import PRIMES


class BooleanTransformer:
    def __init__(self, P: Params, apply_redundancy=True):
        self.P = P
        self.I: InputParams = P.M.I
        self.redundancy = self.I.redundancy
        self.interleaving = self.I.interleaving
        self.apply_redundancy = apply_redundancy
        
        if self.I.encoding == 'binary_embedding':
            self.binary_encoder = BinaryEmbedding(b=self.I.resolution, n=self.I.redundancy)
        else:
            self.binary_encoder = None
    
    def __call__(self, bin_values):
        # Apply tau operation BEFORE redundancy
        if self._has_tau():
            bin_values = self._apply_tau(bin_values)
        
        # needs branch as some encoders handle redundancy internally
        if self.apply_redundancy and self.redundancy > 1:
            bin_values = bin_values.repeat(1, 1, 1, self.redundancy)
        elif self.binary_encoder is not None:
            bin_values = self.binary_encoder.encode_boolean(bin_values)
            
        if self.interleaving:
            bin_values = interleave_features(bin_values, group_size=self.interleaving)
            
        return bin_values.to(torch.uint8)
    
    def _has_tau(self):
        try:
            return self.P.D.tau > 0
        except AttributeError:
            return False

    def _apply_tau(self, x): # key assumption is that tau is applied in last dimension per feature before redundancy is applied
        kqgr = self.P.D
        if kqgr.tau == 0:
            return x

        m, s, f, b = x.shape
        n_identical_bits = kqgr.tau
        assert n_identical_bits <= b, (
            f"tau={n_identical_bits} exceeds bits_per_feature={b}. "
            "Reduce tau or increase resolution."
        )
        ref_idx = torch.randint(0, m, (1,)).item()

        ref = x[ref_idx:ref_idx+1, :, :, :]

        # Helper to assign ref bits to all samples
        def assign_bits(lhs_slice, rhs_slice):
            x[..., lhs_slice] = rhs_slice.expand(m, s, f, n_identical_bits)

        if kqgr.evaluation_mode == 'first':
            assign_bits(slice(0, n_identical_bits), ref[..., :n_identical_bits])
        elif kqgr.evaluation_mode == 'last':
            assign_bits(slice(-n_identical_bits, None), ref[..., -n_identical_bits:])
        elif kqgr.evaluation_mode == 'random':
            # Random indices per feature
            perm = torch.stack([torch.randperm(b, device=x.device)[:n_identical_bits] for _ in range(f)])  # [f, n_identical_bits]

            # Expand for m and s dimensions
            perm_exp = perm[None, None, :, :].expand(m, s, f, n_identical_bits)  # [m, s, f, n_identical_bits]
            ref_exp = ref.expand(m, s, f, b)  # [m, s, f, b]

            # Use take_along_dim for random bits per feature
            x[...] = torch.where(
                torch.zeros_like(x, dtype=torch.bool, device=x.device),
                x,  # unchanged, dummy for broadcasting shape
                x  # just placeholder, overwritten in loop below
            )

            x_new = x.clone()
            for feat in range(f):
                x_new[..., feat, perm[feat]] = ref_exp[..., feat, perm[feat]]
            x[...] = x_new

        return x

class BooleanEncoder:
    def __init__(self, P):
        self.P = P
        self.I: InputParams = P.M.I
        apply_redundancy = (self.I.encoding != 'binary_embedding')
        self.transformer = BooleanTransformer(P, apply_redundancy=apply_redundancy)
    
    def __call__(self, x):
        bin_values = self._encode(x)
        return self.transformer(bin_values)
    
    def _encode(self, values):
        assert values.is_floating_point(), (
            f"BooleanEncoder expects float input (normalized [0,1]). "
            f"Got dtype={values.dtype}. For boolean/integer inputs use BooleanTransformer directly."
        )
        assert torch.max(values) <= 1
        assert torch.min(values) >= 0
        if self.I.encoding == 'base2' or self.I.encoding == 'binary_embedding':
            bin_values = dec2bin(values, self.I.resolution)
        elif self.I.encoding == 'primes':
            bin_values = dec2primes(values, self.I.resolution)
        elif self.I.encoding == 'tally':
            bin_values = dec2tally(values, self.I.resolution)
        else:
            raise ValueError(f"encoding {self.I.encoding} is not an option!")
        return bin_values


def dec2bin(x, bits):
    '''
    Convert decimal to boolean array representation with a fixed number of bits
    Assume input is normalized [0, 1] as float vals
    '''
    x = (x * (2 ** bits - 1)).to(torch.int64)
    mask = (2 ** torch.arange(bits - 1, -1, -1, device=x.device)).to(torch.int64)
    x = (x.unsqueeze(-1).bitwise_and(mask)).ne(0)
    return x


def dec2tally(x, bits):
    '''
    Convert decimal to boolean array representation with a fixed number of bits
    Assume input is normalized [0, 1] as float vals
    '''
    d = (bits * x).round().unsqueeze(-1)
    bit_range = torch.arange(bits, device=x.device).float()
    return bit_range.lt(d)

def load_primes(n):
    if n > 1000:
        raise ValueError(f"Cannot load more than 1000 primes, got {n}")
    return PRIMES[:n]

def dec2primes(x, bits):
    '''
    Convert decimal to boolean array representation using prime number weights.
    Assume input is normalized [0, 1] as float vals.
    Weights are [prime(bits-1), ..., prime(1), 1] where prime(n) is the nth prime.
    '''
    primes = load_primes(bits - 1)  # first (bits-1) primes
    weights = torch.tensor(primes[::-1] + [1], device=x.device, dtype=torch.int64)  # descending
    max_val = weights.sum()
    x = (x * max_val).round().to(torch.int64)
    result = torch.zeros(*x.shape, bits, device=x.device, dtype=torch.bool)
    for i in range(bits):
        result[..., i] = x >= weights[i]
        x = x - result[..., i].long() * weights[i]
    return result

def bin2dec(x, bits, small_endian=False):
    if small_endian:
        mask = 2 ** torch.arange(bits, device=x.device)
    else:
        mask = 2 ** torch.arange(bits - 1, -1, -1, device=x.device)
    mask.to(x.device, x.dtype)
    vals = torch.sum(mask * x, -1).long()
    return vals / (2**bits - 1)


def interleave_features_old(x, group_size=1):  # cant handle uneven b/group
    # for testing x = torch.arange(0, 10).reshape(2, -1).unsqueeze(0).unsqueeze(0)
    # interleave between dimension d mxsxdxb
    # group_size is grouping when interleaving f.ex N=1 → [[0, 1, 2], [3, 4, 5]] → [[0, 3, 1], [4, 2, 5)]
    # group_size is grouping when interleaving f.ex N=2 → [[0, 1, 2], [3, 4, 5]] → [[0, 1, 3], [4, 2, 5)]
    shape = x.shape
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1, group_size)
    x = torch.transpose(x, 2, 3).reshape(shape)
    return x


def interleave_features(x, group_size=1):  # can handle uneven b/group
    # for testing x = torch.arange(0, 10).reshape(2, -1).unsqueeze(0).unsqueeze(0)
    shape = x.shape
    n, m, d, b = shape
    gs = group_size
    num_full = b // gs
    rem = b % gs
    parts = []
    for i in range(num_full):
        for j in range(d):
            part = x[:, :, j:j+1, i*gs: (i+1)*gs]
            parts.append(part)
    if rem > 0:
        for j in range(d):
            part = x[:, :, j:j+1, -rem:]
            parts.append(part)
    interleaved = torch.cat(parts, dim=-1)
    interleaved = interleaved.reshape(shape)
    return interleaved


class min_max_normalization:
    """Stateful min-max normalizer. Stores min_/max_ on first call for later inversion."""
    def __call__(self, data):
        data = data.to(torch.float)
        self.min_ = data.amin(dim=(0, 1), keepdim=True)
        self.max_ = data.amax(dim=(0, 1), keepdim=True)
        return (data - self.min_) / (self.max_ - self.min_)

    def inverse(self, data):
        return data * (self.max_ - self.min_) + self.min_


def standard_normalization(data):
    data = data.to(torch.float)
    # Mean along the samples and steps dimensions
    means = data.mean(dim=(0, 1), keepdim=True)
    # Standard deviation along the samples and steps dimensions
    stds = data.std(dim=(0, 1), keepdim=True)
    return (data - means) / stds


class BinaryEmbedding:
    def __init__(self, b, n):
        self.b = None
        self.n = None
        self.random_boolean_keys = None
        self.set_random_boolean_keys(b, n)

    def _encode_bits(self, bits: torch.Tensor):
        """
        Core encoder operating on boolean bit tensors.

        Args:
            bits: uint8 tensor of shape (B, T, F, b)

        Returns:
            Encoded tensor of shape (B, T, F, n*b)
        """
        batch_size, seq_length, features, b = bits.size()
        assert b == self.b, "Input bit resolution does not match encoder configuration."

        # (B, T, F, 1, b) -> (B, T, F, n, b)
        bits = bits.unsqueeze(3).expand(-1, -1, -1, self.n, -1)

        # XOR with random keys
        encoded = bits ^ self.random_boolean_keys

        # Flatten last two dims
        return encoded.view(batch_size, seq_length, features, -1)

    def encode_float(self, data: torch.Tensor):
        """
        Args:
            data: float tensor in [0,1] of shape (B, T, F)
        """
        bits = dec2bin(data, self.b).to(torch.uint8)
        return self._encode_bits(bits)

    def encode_boolean(self, data: torch.Tensor):
        """
        Args:
            data: boolean / uint8 tensor of shape (B, T, F, b)
        """
        return self._encode_bits(data)

    def set_random_boolean_keys(self, new_b=None, new_n=None):
        self.n = new_n if new_n is not None else self.n
        self.b = new_b if new_b is not None else self.b
        self.random_boolean_keys = torch.randint(
            0, 2, (1, 1, 1, self.n, self.b), dtype=torch.uint8
        )


if __name__ == '__main__':
    bits = 3
    p = torch.randint(0, 2**bits, (3, 2))
    print("Original array:")
    print(p.numpy())

    print("Normalized array:")
    p = min_max_normalization()(p)
    print(p.numpy())

    print("Normalized array times boolean max:")
    print((p * (2**bits - 1)).numpy())

    print("Boolean representation:")
    I = InputParams(bits=bits, chunk_size=bits, encoding='base2', features=p.shape[-1])
    P = Params(M=Params.M(I=I))
    encoder = BooleanEncoder(I)
    boolean_representation = encoder(p)
    print(boolean_representation.to(torch.int).numpy())

    ##########################################################
    encoder = BinaryEmbedding(b=4, n=3)
    batch_size = 2
    s = 3
    features = 2
    x = torch.randint(0, 10, (batch_size, s, features,),
                      dtype=torch.float) / 10

    encoded_tensors = encoder.encode_float(x)
    print("Input Tensor:\n", x.numpy())
    print("Encoded Tensor Shape:", encoded_tensors.shape)
    print("Encoded Tensor:\n", encoded_tensors.to(torch.int).numpy())
