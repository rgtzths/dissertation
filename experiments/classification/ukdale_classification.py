import datetime
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import randint, loguniform

from run_experiment import run

import sys
sys.path.insert(1, "../../classification_models")
from gacd import GACD
from seq2point import Seq2Point
from gru_dwt import GRU_DWT

def run_experiment():

    #Experiment Definition
    experiment = {
        "fridge" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                "GACD" :GACD( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "appliances" : {
                        "fridge" : {
                            "dwt_timewindow" : 12,
                            "dwt_overlap" : 10,
                            "examples_overlap" : 0,
                            "examples_timewindow" : 300,
                            "epochs" : 1,
                            "batch_size" : 1024,
                            "wavelet": 'bior2.2',
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
                "GRU_DWT" : GRU_DWT( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "appliances" : {
                        "fridge" : {
                            "dwt_timewindow" : 12,
                            "dwt_overlap" : 10,
                            "examples_overlap" : 0,
                            "examples_timewindow" : 300,
                            "epochs" : 1,
                            "batch_size" : 1024,
                            "wavelet": 'bior2.2',
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
                "Seq2Point" : Seq2Point({'n_epochs':1,'batch_size':1024, "training_results_path" : "/home/user/thesis_results/history/"})
            },
            "model_path" : "/home/user/thesis_results/models/",
            "timestep" : 2,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2013, 5, 22),
                            "end" : datetime.datetime(2013, 5, 23)
                            #"end" : datetime.datetime(2013, 7, 1)
                        },
                        "house_2" : {
                            "beginning" : datetime.datetime(2013, 5, 23),
                            "end" : datetime.datetime(2013, 5, 24)
                            #"end" : datetime.datetime(2013, 7, 1)
                        }
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        "house_5" : {
                            "beginning" : datetime.datetime(2014, 6, 30),
                            #"end" : datetime.datetime(2014, 7, 9)
                            "end" : datetime.datetime(2014, 6, 30, 12)

                        }
                    }
                }
            }
        },
        "microwave" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                "GACD" :GACD( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "appliances" : {
                        "microwave" : {
                            "dwt_timewindow" : 12,
                            "dwt_overlap" : 10,
                            "examples_overlap" : 0,
                            "examples_timewindow" : 300,
                            "epochs" : 1,
                            "batch_size" : 1024,
                            "wavelet": 'bior2.2',
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
                "GRU_DWT" : GRU_DWT( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "appliances" : {
                        "microwave" : {
                            "dwt_timewindow" : 12,
                            "dwt_overlap" : 10,
                            "examples_overlap" : 0,
                            "examples_timewindow" : 300,
                            "epochs" : 1,
                            "batch_size" : 1024,
                            "wavelet": 'bior2.2',
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
                #"Seq2Point" : Seq2Point({'n_epochs':1,'batch_size':1024, "training_results_path" : "/home/user/thesis_results/history/"})
            },
            "model_path" : "/home/user/thesis_results/models/",
            "timestep" : 2,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2013, 5, 23),
                            "end" : datetime.datetime(2013, 5, 25)
                            #"end" : datetime.datetime(2013, 7, 1)
                        },
                        "house_2" : {
                            "beginning" : datetime.datetime(2013, 5, 23),
                            "end" : datetime.datetime(2013, 5, 25)
                            #"end" : datetime.datetime(2013, 7, 1)
                        }
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        "house_5" : {
                            "beginning" : datetime.datetime(2014, 6, 30),
                            #"end" : datetime.datetime(2014, 7, 9)
                            "end" : datetime.datetime(2014, 7, 3)

                        }
                    }
                }
            }
        },
        "dish_washer" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                "GACD" :GACD( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "appliances" : {
                        "dish_washer" : {
                            "dwt_timewindow" : 12,
                            "dwt_overlap" : 10,
                            "examples_overlap" : 0,
                            "examples_timewindow" : 300,
                            "epochs" : 1,
                            "batch_size" : 1024,
                            "wavelet": 'bior2.2',
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
                "GRU_DWT" : GRU_DWT( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "appliances" : {
                        "dish_washer" : {
                            "dwt_timewindow" : 12,
                            "dwt_overlap" : 10,
                            "examples_overlap" : 0,
                            "examples_timewindow" : 300,
                            "epochs" : 1,
                            "batch_size" : 1024,
                            "wavelet": 'bior2.2',
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
                #"Seq2Point" : Seq2Point({'n_epochs':1,'batch_size':1024, "training_results_path" : "/home/user/thesis_results/history/"})
            },
            "model_path" : "/home/user/thesis_results/models/",
            "timestep" : 2,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2013, 5, 20),
                            "end" : datetime.datetime(2013, 5, 22)
                            #"end" : datetime.datetime(2013, 7, 1)
                        },
                        "house_2" : {
                            "beginning" : datetime.datetime(2013, 5, 23),
                            "end" : datetime.datetime(2013, 5, 24)
                            #"end" : datetime.datetime(2013, 7, 1)
                        }
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        "house_5" : {
                            "beginning" : datetime.datetime(2014, 6, 30),
                            #"end" : datetime.datetime(2014, 7, 9)
                            "end" : datetime.datetime(2014, 6, 30, 12)

                        }
                    }
                }
            }
        }
    }

    run(experiment)

if __name__ == "__main__":
    run_experiment()
