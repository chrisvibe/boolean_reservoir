from project.boolean_reservoir.code.utils.utils import save_grid_search_results
from project.boolean_reservoir.code.kq_and_gr_metric import calc_kernel_quality_and_generalization_rank

if __name__ == '__main__':
    paths = list()
    # paths.append('config/temporal/kq_and_gr/test.yaml')

    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/heterogeneous_stochastic.yaml')

    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/heterogeneous_stochastic.yaml')

    paths.append('config/temporal/kq_and_gr/vary_delay/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/vary_delay/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/vary_delay/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/vary_delay/heterogeneous_stochastic.yaml')

    # generate data
    for path in paths:
        print(path)
        df, P = calc_kernel_quality_and_generalization_rank(path)
        data_file_path = P.L.out_path / 'log.yaml'
        save_grid_search_results(df, data_file_path)
