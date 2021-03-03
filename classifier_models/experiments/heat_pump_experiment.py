import sys
sys.path.insert(1, "..")

import dataset_loader
from lstm import LSTM_RNN
from gru import GRU_RNN
import datetime

from run_experiment import run_experiment

experiment = {
    "heatpump" : {
        "methods" : {
            "LSTM_1_0" : LSTM_RNN( {
                        "timeframe" : 1,
                        "timestep" : 2,
                        "overlap" : 0.5,
                        "interpolate" : "average",
                        "epochs" : 400,
                        "input_size" : 0,
                        "model_name" : "LSTM_1_0"
            }),
            "LSTM_5_0" : LSTM_RNN( {
                        "timeframe" : 5,
                        "timestep" : 2,
                        "overlap" : 0.5,
                        "interpolate" : "average",
                        "epochs" : 400,
                        "input_size" : 0,
                        "model_name" : "LSTM_5_0"
            }),
            "LSTM_10_0" : LSTM_RNN( {
                        "timeframe" : 10,
                        "timestep" : 2,
                        "overlap" : 0.5,
                        "interpolate" : "average",
                        "epochs" : 400,
                        "input_size" : 0,
                        "model_name" : "LSTM_10_0"
            }),
            "LSTM_30_0" : LSTM_RNN( {
                        "timeframe" : 30,
                        "timestep" : 2,
                        "overlap" : 0.5,
                        "interpolate" : "average",
                        "epochs" : 400,
                        "input_size" : 0,
                        "model_name" : "LSTM_30_0"
            }),
            "LSTM_60_0" : LSTM_RNN( {
                        "timeframe" : 60,
                        "timestep" : 2,
                        "overlap" : 0.5,
                        "interpolate" : "average",
                        "epochs" : 400,
                        "input_size" : 0,
                        "model_name" : "LSTM_60_0"
            }),
            "LSTM_120_0" : LSTM_RNN( {
                        "timeframe" : 10,
                        "timestep" : 2,
                        "overlap" : 0.5,
                        "interpolate" : "average",
                        "epochs" : 400,
                        "input_size" : 0,
                        "model_name" : "LSTM_120_0"
            }),
            "GRU_1_0" : GRU_RNN( {
                        "timeframe" : 1,
                        "timestep" : 2,
                        "overlap" : 0.5,
                        "interpolate" : "average",
                        "epochs" : 400,
                        "input_size" : 0,
                        "model_name" : "GRU_1_0"
            }),
            "GRU_5_0" : GRU_RNN( {
                        "timeframe" : 5,
                        "timestep" : 2,
                        "overlap" : 0.5,
                        "interpolate" : "average",
                        "epochs" : 400,
                        "input_size" : 0,
                        "model_name" : "GRU_5_0"
            }),
            "GRU_10_0" : GRU_RNN( {
                        "timeframe" : 10,
                        "timestep" : 2,
                        "overlap" : 0.5,
                        "interpolate" : "average",
                        "epochs" : 400,
                        "input_size" : 0,
                        "model_name" : "GRU_10_0"
            }),
            "GRU_30_0" : GRU_RNN( {
                        "timeframe" : 30,
                        "timestep" : 2,
                        "overlap" : 0.5,
                        "interpolate" : "average",
                        "epochs" : 400,
                        "input_size" : 0,
                        "model_name" : "GRU_30_0"
            }),
            "GRU_60_0" : GRU_RNN( {
                        "timeframe" : 60,
                        "timestep" : 2,
                        "overlap" : 0.5,
                        "interpolate" : "average",
                        "epochs" : 400,
                        "input_size" : 0,
                        "model_name" : "GRU_60_0"
            }),
            "GRU_120_0" : GRU_RNN( {
                        "timeframe" : 10,
                        "timestep" : 2,
                        "overlap" : 0.5,
                        "interpolate" : "average",
                        "epochs" : 400,
                        "input_size" : 0,
                        "model_name" : "GRU_120_0"
            }),
        },
        "model_path" : "./models/",
        "train" : {
            "avEiro" : {
                "location" : "../../../datasets/avEiro_classification/",
                "houses" : {
                    "house_1" : {
                        "beginning" : None,
                        "end" : datetime.datetime(2021, 1, 31)
                    }
                }
            },
        },
        "test" : {
            "avEiro" : {
                "location" : "../../../datasets/avEiro_classification/",
                "houses" : {
                    "house_1" : {
                        "beginning" : datetime.datetime(2021, 1, 31),
                        "end" : None
                    }
                }
            }
        }
    }
}

run_experiment(experiment)

