import os
from datetime import datetime


def create_experiment_folders(dataset_folder, model_name, post_fix=''):
    experiment_dir = 'experiments'
    # create folder for experiments with current dataset
    dataset_path = create_folder(os.path.join(experiment_dir, dataset_folder))
    # create current experiment folder
    current_time = datetime.now().strftime('%d%m%Y-%H%M%S')
    experiment_folder_name = '{0}-{1}-{2}'.format(model_name,
                                                  post_fix,
                                                  current_time)
    experiment_path = create_folder(
        os.path.join(
            dataset_path,
            experiment_folder_name))
    # create folder for logging
    log_path = create_folder(os.path.join(experiment_path, 'logs'))
    # create folder for tensorboard
    tensorboard_path = create_folder(
        os.path.join(experiment_path, 'tensorboard'))
    # create folder for saving model
    save_path = create_folder(os.path.join(experiment_path, 'saved_model'))

    return {'experiment': experiment_path,
            'log': log_path,
            'tensorboard': tensorboard_path,
            'save': save_path}


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

    return path


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
