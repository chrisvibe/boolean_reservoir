from project.boolean_reservoir.code.train_model import train_single_model
from project.boolean_reservoir.code.visualization import plot_train_history, plot_dynamics_history, plot_activity_trace
from project.path_integration.code.visualization import plot_many_things

if __name__ == '__main__':
    # p, model, dataset, history = train_single_model(yaml_or_checkpoint_path='/tmp/pi/continuous/1.yaml', dataset_init=d(), accuracy=a().accuracy)
    # # plot_many_things(model, dataset, history)

    # '''
    # issue: accuracy much higher run on cpu than gpu (exported yaml config and re-ran)
    # random seeds are ofc different on cpu vs gpu, but the cpu accuracy (~0.22) is almost an outlier in gpu distribution (~0.5)
    # TODO bug found and squashed (velocity not reset between samples), but need to verify this is gone
    # '''


    # p, model, dataset, history = train_single_model(yaml_or_checkpoint_path='/tmp/pi/discrete/1.yaml')
    # p, model, dataset, history = train_single_model(yaml_or_checkpoint_path='/tmp/pi/continuous/1.yaml')
    # p, model, dataset, history = train_single_model(yaml_or_checkpoint_path='/tmp/pi/continuous/006_accuracy.yaml')
    # p, model, dataset, history = train_single_model(yaml_or_checkpoint_path='/tmp/pi/continuous/006_accuracy2.yaml')
    # plot_many_things(model, dataset, history)

    # p, model, dataset, history = train_single_model(yaml_or_checkpoint_path='/code/config/path_integration/2D/single_run/good_model.yaml')
    p, model, dataset, history = train_single_model(yaml_or_checkpoint_path='/code/config/path_integration/2D/single_run/good_model2.yaml')