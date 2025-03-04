
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd
from collections import Counter
from boolean_reservoir.graphs import generate_adjacency_matrix
matplotlib.use('Agg')

def visualize_in_degree_distribution_from_adjacency_matrix_func(path, file_name_append, samples, adjacency_matrix_func, **func_params):
    all_degrees = []
    adj_matrix_gen = (adjacency_matrix_func(**func_params) for _ in range(samples))
    for adj_matrix in adj_matrix_gen:
        degrees = adj_matrix.sum(axis=0).tolist()
        all_degrees.extend(degrees)
    
    degree_counts = Counter(all_degrees)
    degrees = list(degree_counts.keys())
    frequencies = list(degree_counts.values())
    plt.bar(degrees, frequencies, edgecolor='black')
    plt.title('Frequency Plot of Node Degrees')
    plt.xlabel('Degree')
    plt.xticks(degrees)
    plt.ylabel('Frequency')
    
    path = Path(path) / 'visualizations/in_degree_distributions'
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"adjacency_matrix_in_degree_distribution_{file_name_append}.png"
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def visualize_in_degree_distribution_from_adjacency_matrix_func_vary_parameter(path, file_name_append, samples, parameter, parameter_iterable, adjacency_matrix_func, **func_params):
    data = []
    for parameter_val in parameter_iterable:
        func_params_i = func_params.copy()
        func_params_i[parameter] = parameter_val
        for _ in range(samples):
            degrees = adjacency_matrix_func(**func_params_i).sum(axis=0)
            degrees = np.atleast_1d(degrees)
            for degree in degrees:
                data.append((parameter_val, degree))
    
    df = pd.DataFrame(data, columns=['parameter', 'degree'])
    
    stats_df = df.groupby('parameter')['degree'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=stats_df, x='parameter', y='mean', label='Mean', linestyle='-')
    sns.lineplot(data=stats_df, x='parameter', y='median', label='Median', linestyle='--')
    sns.lineplot(data=stats_df, x='parameter', y='std', label='Standard Deviation', linestyle='-.')
    sns.lineplot(data=stats_df, x='parameter', y='min', label='Min', linestyle=':')
    sns.lineplot(data=stats_df, x='parameter', y='max', label='Max', linestyle='-')
    plt.title(f'Statistical Properties of {parameter.capitalize()} Over Time ({file_name_append})')
    plt.xlabel(parameter.capitalize())
    plt.ylabel('In-degree Value')
    plt.legend()
    
    path = Path(path) / 'visualizations/in_degree_distributions'
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"adjacency_matrix_in_degree_distribution_statistical_properties_over_time_{parameter}_{file_name_append}.png"
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    ###################################### compare old incorrect adjacency matrix function with new #############################################################################

    from boolean_reservoir.graphs import old_generate_adjacency_matrix, generate_adjacency_matrix
    samples = 100
    func_params = dict()
    func_params['n_nodes'] = 100
    func_params['k_min'] = 0
    func_params['k_avg'] = 3
    func_params['k_max'] = 6
    func_params['self_loops'] = None 

    # # check for expected normal distribution 
    visualize_in_degree_distribution_from_adjacency_matrix_func('/out', 'old', samples, old_generate_adjacency_matrix, **func_params)
    visualize_in_degree_distribution_from_adjacency_matrix_func('/out', 'new', samples, generate_adjacency_matrix, **func_params)

    # # check that properties are invariant to n_nodes
    samples = 20
    parameter = 'n_nodes'
    parameter_iterable = range(100, 1000 + 1, 20) 
    visualize_in_degree_distribution_from_adjacency_matrix_func_vary_parameter('/out', 'old', samples, parameter, parameter_iterable, old_generate_adjacency_matrix, **func_params)
    visualize_in_degree_distribution_from_adjacency_matrix_func_vary_parameter('/out', 'new', samples, parameter, parameter_iterable, generate_adjacency_matrix, **func_params)

    # check that std is largest at self_loops = 0.5 as it has most freedom
    # should be lower at p=1 as it spreads out in_degree and p=0 as it limits number of edges
    samples = 100
    parameter = 'self_loops'
    parameter_iterable = [i/100 for i in range(100+1)]
    visualize_in_degree_distribution_from_adjacency_matrix_func_vary_parameter('/out', 'old', samples, parameter, parameter_iterable, old_generate_adjacency_matrix, **func_params)
    visualize_in_degree_distribution_from_adjacency_matrix_func_vary_parameter('/out', 'new', samples, parameter, parameter_iterable, generate_adjacency_matrix, **func_params)

    ###################################### verify that underlying sub-routines are correct and roughly check run-time scaling #############################################################################

    from boolean_reservoir.graphs import randomly_distribute_pigeons_to_holes_with_capacity_n_rows_at_a_time, randomly_distribute_pigeons_to_holes_with_capacity_1_row_at_a_time, randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick, randomly_project_1d_to_2d_avoiding_diagonal
    from time import time
    func_params = dict()
    parameter = 'pigeons'

    print('\nmany nodes/holes...')
    func_params['holes'] = 10
    func_params['capacity'] = 3
    samples = 1000
    parameter_iterable = range(1, func_params['holes'] * func_params['capacity'] + 1, 1) 
    f_dimension_trick = lambda **func_params: randomly_project_1d_to_2d_avoiding_diagonal(randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(**func_params))
    f_1_row = lambda **func_params: randomly_project_1d_to_2d_avoiding_diagonal(randomly_distribute_pigeons_to_holes_with_capacity_1_row_at_a_time(**func_params))
    f_n_row = lambda **func_params: randomly_project_1d_to_2d_avoiding_diagonal(randomly_distribute_pigeons_to_holes_with_capacity_n_rows_at_a_time(**func_params))
    start = time()
    visualize_in_degree_distribution_from_adjacency_matrix_func_vary_parameter('/out', f'n_{func_params['holes']}_pidgeons_dimension_trick', samples, parameter, parameter_iterable, f_dimension_trick, **func_params)
    end = time()
    print(end - start)
    start = time()
    visualize_in_degree_distribution_from_adjacency_matrix_func_vary_parameter('/out', f'n_{func_params['holes']}_pidgeons_1_row', samples, parameter, parameter_iterable, f_1_row, **func_params)
    end = time()
    print(end - start)
    start = time()
    visualize_in_degree_distribution_from_adjacency_matrix_func_vary_parameter('/out', f'n_{func_params['holes']}_pidgeons_n_rows', samples, parameter, parameter_iterable, f_n_row, **func_params)
    end = time()
    print(end - start)

    print('\nmany nodes/holes...')
    func_params['holes'] = 3000
    func_params['capacity'] = 3
    samples = 10
    parameter_iterable = range(1000, func_params['holes'] * func_params['capacity'] + 1, 100) 
    f_dimension_trick = lambda **func_params: randomly_project_1d_to_2d_avoiding_diagonal(randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(**func_params))
    f_1_row = lambda **func_params: randomly_project_1d_to_2d_avoiding_diagonal(randomly_distribute_pigeons_to_holes_with_capacity_1_row_at_a_time(**func_params))
    f_n_row = lambda **func_params: randomly_project_1d_to_2d_avoiding_diagonal(randomly_distribute_pigeons_to_holes_with_capacity_n_rows_at_a_time(**func_params))
    start = time()
    visualize_in_degree_distribution_from_adjacency_matrix_func_vary_parameter('/out', f'n_{func_params['holes']}_pidgeons_dimension_trick', samples, parameter, parameter_iterable, f_dimension_trick, **func_params)
    end = time()
    print(end - start)
    start = time()
    visualize_in_degree_distribution_from_adjacency_matrix_func_vary_parameter('/out', f'n_{func_params['holes']}_pidgeons_1_row', samples, parameter, parameter_iterable, f_1_row, **func_params)
    end = time()
    print(end - start)
    start = time()
    visualize_in_degree_distribution_from_adjacency_matrix_func_vary_parameter('/out', f'n_{func_params['holes']}_pidgeons_n_rows', samples, parameter, parameter_iterable, f_n_row, **func_params)
    end = time()
    print(end - start)