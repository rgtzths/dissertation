import datetime
import warnings
warnings.filterwarnings("ignore")

from lstm import LSTM_RNN
from gru import GRU_RNN
from run_experiment import run
from gradient import GradientBoosting
from cnn import CNN
from svm import SVM
def run_experiment():

    #Experiment Definition
    experiment = {
        "heatpump" : {
            "methods" : {
                "LSTM_10_0" : LSTM_RNN( {
                            "timeframe" : 10,
                            "timestep" : 2,
                            "overlap" : 0.5,
                            "interpolate" : "average",
                            "epochs" : 300,
                            "input_size" : 0,
                            "model_name" : "LSTM_10_0"
                }),
                "GRU_10_0" : GRU_RNN( {
                            "timeframe" : 10,
                            "timestep" : 2,
                            "overlap" : 0.5,
                            "interpolate" : "average",
                            "epochs" : 300,
                            "input_size" : 0,
                            "model_name" : "GRU_10_0"
                }),
                "Gradient_10_0" : GradientBoosting( {
                            "timeframe" : 10,
                            "timestep" : 2,
                            "overlap" : 0.5,
                            "interpolate" : "average",
                            "epochs" : 300,
                            "model_name" : "Gradient_10"
                }),
                "CNN_10_0" : CNN( {
                            "timeframe" : 10,
                            "timestep" : 2,
                            "overlap" : 0.5,
                            "interpolate" : "average",
                            "epochs" : 300,
                            "model_name" : "CNN_10"
                }),
                "SVM" : SVM({})
            },
            "model_path" : "./models/",
            "train" : {
                "ukdale" : {
                    "location" : "../../datasets/avEiro_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2020, 10, 1),
                            "end" :  datetime.datetime(2020, 10, 1, 1)
                        }
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../datasets/avEiro_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2020, 10, 1, 1),
                            "end" : datetime.datetime(2020, 10, 1, 3, 3)
                        }
                    }
                }
            }
        }
    }

    run(experiment)

if __name__ == "__main__":
    run_experiment()