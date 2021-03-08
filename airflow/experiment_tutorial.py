from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
import sys
sys.path.insert(1, "/home/user/Desktop/thesis/regression_models/experiments/")

import dag_boiler_experiment

default_args = {
    'owner' : "admin",
}

@dag(default_args=default_args, schedule_interval=None, start_date=days_ago(2), tags=['example', 'experiment'])
def experiment_tutorial():
    '''
    ### Experiment Tutorial Documentation
    - This DAG runs multiple experiments
    '''

    @task()
    def experiment():
        '''
        #### Experiment Task
        - Runs a predefined experiment.
        '''
        dag_boiler_experiment.run_experiment()



    experiment()

experiment_tutorial_dag = experiment_tutorial()