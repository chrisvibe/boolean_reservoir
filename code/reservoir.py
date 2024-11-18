import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone


class BooleanReservoir(nn.Module):
    def __init__(self, bits_per_feature, n_features, reservoir_size, output_size, lut_length, device, primes, record=True, out_path='/out', seed=42, max_history_buffer_size=10000):
        super(BooleanReservoir, self).__init__()
        torch.manual_seed(seed)  # Seed for reproducibility
        self.bits_per_feature = bits_per_feature
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.device = device

        self.out_path = Path(out_path)
        self.folders = ['reservoir_history']
        for f in self.folders:
            p = self.out_path / f 
            if not p.exists():
                p.mkdir(parents=True)
        self.record = record
        self.max_history_buffer_size = max_history_buffer_size
        self.history_buffer = list() 
        self.history_buffer_file_count = 0
        
        # Create a list of random lookup tables for each node in the reservoir
        lut_list = [torch.randint(0, 2, (2 ** lut_length,), dtype=torch.bool) for _ in range(reservoir_size)]
        self.lut_tensor = torch.stack(lut_list).to(device, dtype=torch.int)

        # Initialize reservoir states
        self.initial_reservoir = torch.randint(0, 2, (reservoir_size,), dtype=torch.bool).to(device)

        # Initialize graph (adjacency matrix)
        self.W_reservoir = torch.randint(0, 2, (reservoir_size, reservoir_size), dtype=torch.bool).to(device)

        # Dense readout layer
        self.readout = nn.Linear(reservoir_size, output_size)

        # prime number sums are unique assuming no repetition and is used to label each node -> unique state combination indeces
        self.primes = torch.tensor(primes[:reservoir_size], dtype=torch.int) # start from 3

        # precompute for later...
        self.node_indeces = torch.arange(self.reservoir_size).to(self.device)

        # Preselect which reservoir nodes will be overwritten by the input data
        # TODO confine features to certain areas of the reservoir??? Now they are mixing...
        self.input_nodes = torch.randperm(reservoir_size)[:bits_per_feature * n_features].to(device)


    def sample_init(self):
        self.reset_reservoir()
        # TODO warmup reservoir?

    def reset_reservoir(self):
        # TODO reset output layer too?
        self.reservoir = self.initial_reservoir.clone()

    def save_record(self, record):
        self.history_buffer.append(record)
        if len(self.history_buffer) >= self.max_history_buffer_size:
            self.flush_history()
    
    def flush_history(self):
        if self.record and self.history_buffer:
            time = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M")
            f = str(self.history_buffer_file_count) + '_' + time + '.csv'
            df = pd.DataFrame(self.history_buffer)
            df.to_csv(self.out_path / 'reservoir_history' / f, index=False) # TODO consider np.savez_compressed
            self.history_buffer = list()
            self.history_buffer_file_count += 1

    def forward(self, x):
        m, s, d, b = x.shape
        outputs = []
        '''
        Accepts the decomposed velocities encoded in boolean format. In 2d: x = dx, dy
        Since each series of velocities corresponds to a single coordinate label the shape of x is: mxsxd
        m is the number of samples
        s is the number of steps
        d is the number of dimensions
        b is the number of bits used for the boolean encoding
        the output is then mxdxb

        1. input the encoded velocity data in s steps
        2. the reservoir should is not hopefully able to represent the integral of these steps
        3. readout interprets the reservoir and outputs the integral; the final position coordinate

        TODO order of operations below: input step vs state index calculation!! good?
        '''
        
        for i in range(m):
            self.sample_init()
            # INPUT LAYER
            for j in range(s):
                # Overwrite specific reservoir nodes with input data
                # TODO we need bits_per_feature * n_features bits in the reservoir for this approach!!!
                self.reservoir[self.input_nodes] = x[i][j].flatten()

                # Record states
                if self.record:
                    record = dict()
                    record['sample'] = i 
                    record['reservoir_states'] = (torch.clone(self.reservoir).to(torch.int).numpy())
                    self.save_record(record)

                # Calculate an index for each reservoir state
                state_idx = (self.reservoir * self.primes * self.W_reservoir).sum(dim=1)

                # RESERVOIR LAYER
                # Use LUT to update the reservoir nodes in parallel
                perturbed_reservoir = (self.lut_tensor[self.node_indeces, state_idx]).bool()
            
                # Update the reservoir
                self.reservoir = perturbed_reservoir
            
            # READOUT LAYER
            output = self.readout(self.reservoir.float())
            outputs.append(output)
        
        return torch.stack(outputs)



class ThreeLayerPerceptron2(nn.Module):
    def __init__(self, bits_per_feature, n_features):
        super(ThreeLayerPerceptron, self).__init__()
        pass
        
        '''
        # bits_per_features
        boolean representation of feature

        # n_features
        number of features

        # example init
        model = ThreeLayerPerceptron(5, 2)

        # example forward
        y_hat = model(x)
        x has dimension torch.Size([50, 1000, 2, 5]) and is a boolean type
        y hat is torch.Size([50, 2]) and is a float
        note that 50 is the batch size in this case
        each x sample has 1000 parts which predict a single (x, y) float coordinate
        '''


class ThreeLayerPerceptron(nn.Module):
    def __init__(self, bits_per_feature, n_features, n_steps):
        super(ThreeLayerPerceptron, self).__init__()

        # Store the number of features, bits per feature, and steps
        self.bits_per_feature = bits_per_feature
        self.n_features = n_features
        input_size = n_steps * n_features * bits_per_feature
        
        # Define layers with the computed input size
        self.layer1 = nn.Linear(input_size, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 2)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Ensure input is float type
        x = x.float()
        
        # Flatten the input tensor except for the batch dimension
        x = x.view(x.size(0), -1)  # Shape: [batch_size, variable_size * n_features * bits_per_feature]

        # Pass through the first layer
        x = self.layer1(x)
        x = self.relu(x)
        
        # Pass through the second layer
        x = self.layer2(x)
        x = self.relu(x)
        
        # Pass through the third layer
        y_hat = self.layer3(x)
        
        return y_hat
 

