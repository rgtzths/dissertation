from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
import sys
sys.path.insert(1, "/home/user/Desktop/ZiTh0s/Uni/Tese/codigo/thesis/airflow/regression_experiments/")

import boiler_experiment
import car_charger_experiment
import dish_washer_experiment
import fridge_experiment
import heat_pump_experiment
import kettle_experiment
import microwave_experiment
import oven_experiment
import toaster_experiment
import washing_machine_experiment

default_args = {
    'owner' : "admin",
}

@dag(default_args=default_args, schedule_interval=None, start_date=days_ago(2), tags=['example', 'experiment'])
def regression_experiment():
    '''
    ### Regression Experiments Documentation
    - This DAG runs multiple experiments for NILM regression
    '''

    @task()
    def boiler():
        '''
        #### Boiler Task
        - Runs a predefined experiment on the boiler appliance.
        '''
        boiler_experiment.run_experiment()

    @task()
    def car_charger():
        '''
        #### Car charger Task
        - Runs a predefined experiment on the car charger appliance.
        '''
        car_charger_experiment.run_experiment()
    
    @task()
    def dish_wahser():
        '''
        #### Dish Washer Task
        - Runs a predefined experiment on the dish washer appliance.
        '''
        dish_washer_experiment.run_experiment()

    @task()
    def fridge():
        '''
        #### Fridge Task
        - Runs a predefined experiment on the fridge appliance.
        '''
        fridge_experiment.run_experiment()

    @task()
    def heat_pump():
        '''
        #### Heat Pump Task
        - Runs a predefined experiment on the heat pump appliance.
        '''
        heat_pump_experiment.run_experiment()

    @task()
    def kettle():
        '''
        #### Kettle Task
        - Runs a predefined experiment on the Kettle appliance.
        '''
        kettle_experiment.run_experiment()
    
    @task()
    def microwave():
        '''
        #### Microwave Task
        - Runs a predefined experiment on the microwave appliance.
        '''
        microwave_experiment.run_experiment()
    
    @task()
    def oven():
        '''
        #### Oven Task
        - Runs a predefined experiment on the Oven appliance.
        '''
        oven_experiment.run_experiment()
    
    @task()
    def toaster():
        '''
        #### Toaster Task
        - Runs a predefined experiment on the Toaster appliance.
        '''
        toaster_experiment.run_experiment()
    
    @task()
    def washing_machine():
        '''
        #### Washing Machine Task
        - Runs a predefined experiment on the washing machine appliance.
        '''
        washing_machine_experiment.run_experiment()
    
    boiler()
    car_charger()
    dish_wahser()
    fridge()
    heat_pump()
    kettle()
    microwave()
    oven()
    toaster()
    washing_machine()

regression_experiment_dag = regression_experiment()