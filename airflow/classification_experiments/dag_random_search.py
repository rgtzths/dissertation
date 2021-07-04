from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
import sys
sys.path.insert(1, "/home/user/Desktop/ZiTh0s/Uni/Tese/codigo/thesis/airflow/classification_experiments/")

import random_search


default_args = {
    'owner' : "admin",
}

dwt_timewindows = [8, 24, 48, 60]
examples_timewindows = [300], 600, 1800, 3600]
wavelets = ['bior2.2', 'coif2', 'db2', 'rbio2.2']

@dag(default_args=default_args, schedule_interval="Once", start_date=days_ago(2), tags=['randomsearch'])
def classification_experiment():
    '''
    ### Classification Experiments Documentation
    - This DAG runs multiple experiments for NILM classification
    '''

    @task()
    def task1(dwt_timewindow, dwt_overlap, examples_overlap, examples_timewindow, wavelet):
        random_search.run_experiment(dwt_timewindow, dwt_overlap, examples_overlap, examples_timewindow, wavelet)

    for dwt_timewindow in dwt_timewindows:
        for example_timewindow in example_timewindows:
            for wavelt in wavelets:
                task1(dwt_timewindow, dwt_overlap[dwt_timewindow], example_overla[example_timewindow], example_timewindow, wavelet)
        

classification_experiment_dag = classification_experiment()