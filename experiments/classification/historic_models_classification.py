import datetime
import warnings
warnings.filterwarnings("ignore")

from run_experiment import run

import sys
sys.path.insert(1, "../../classification_models/model2")
sys.path.insert(1, "../../utils")
sys.path.insert(1, "../../feature_extractors")

from deep_gru import DeepGRU
from resnet import ResNet
#from mlp import MLP

base_path= "/home/rteixeira/historic_classification_ukdale/model2/"
epochs = 300
batch_size = 512
sequence_length = 60
timestep = 6 

def run_experiment():

    #Experiment Definition
    experiment = {
        "fridge" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'active')],
            "use_activations" : True,
            "appliances_params" : {
                    "min_off_time" : 12,
                    "min_on_time" : 60,
                    "number_of_activation_padding": 100,
                    "min_on_power" : 50
            },
            "methods" : {
                "DeepGRU" : DeepGRU({ 
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/DeepGRU/",
                    "results_folder" : base_path + "results/DeepGRU/",
                    "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                    "plots_folder" : base_path + "plots/DeepGRU/",
                    "appliances" : {
                        "fridge" : {
                            'timewindow' : timestep*sequence_length,
                            'timestep' : timestep,
                            'overlap' :  timestep*sequence_length - timestep,
                            'n_nodes' : 90,
                            'epochs' : epochs,
                            'on_treshold' : 50,
                        }
                    },
                }),
                "ResNet" : ResNet({ 
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/ResNet/",
                    "results_folder" : base_path + "results/ResNet/",
                    "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                    "plots_folder" : base_path + "plots/ResNet/",
                    "appliances" : {
                        "fridge" : {
                            'timewindow' : timestep*sequence_length,
                            'timestep' : timestep,
                            'overlap' :  timestep*sequence_length - timestep,
                            'n_nodes' : 64,
                            'epochs' : epochs,
                            'on_treshold' : 50,
                        }
                    },
                }),
                #"MLP" : MLP({ 
                #    'verbose' : 2,
                #    "training_history_folder" : base_path + "history/MLP/",
                #    "results_folder" : base_path + "results/MLP/",
                #    "checkpoint_folder" : base_path + "temp_weights/MLP/",
                #    "plots_folder" : base_path + "plots/MLP/",
                #    "appliances" : {
                #        "fridge" : {
                #            'timewindow' : timestep*sequence_length,
                #            'timestep' : timestep,
                #            'overlap' :  timestep*sequence_length - timestep,
                #            'n_nodes' : 256,
                #            'epochs' : epochs,
                #            'on_treshold' : 50,
                #            'use_dwt' : True,
                #            'wavelet' : 'db4'
                #        }
                #    },
                #}),
                #"MLP_Raw" : MLP({ 
                #    'model_name' : 'MLP_Raw',
                #    'verbose' : 2,
                #    "training_history_folder" : base_path + "history/MLP_Raw/",
                #    "results_folder" : base_path + "results/MLP_Raw/",
                #    "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                #    "plots_folder" : base_path + "plots/MLP_Raw/",
                #    "appliances" : {
                #        "fridge" : {
                #            'timewindow' : timestep*sequence_length,
                #            'timestep' : timestep,
                #            'overlap' :  timestep*sequence_length - timestep,
                #            'n_nodes' : 256,
                #            'epochs' : epochs,
                #            'on_treshold' : 50,
                #            'use_dwt' : False,
                #        }
                #    },
                #}),
            },
            "model_path" : base_path+"models/",
            "timestep" : 6,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        1 : [(datetime.datetime(2013, 5, 1), datetime.datetime(2013, 5, 15))],#datetime.datetime(2013, 5, 15)
                    }
                },
            },
            "cross_validation" :{
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        2 : [(datetime.datetime(2013, 7, 1),datetime.datetime(2013, 7, 8))],#datetime.datetime(2013, 7, 8))
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        5 : [(datetime.datetime(2014, 9, 30), datetime.datetime(2014, 10, 30))]#datetime.datetime(2014, 10, 30))
                    }
                }
            }
        },
        "kettle" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'active')],
            "use_activations" : True,
            "appliances_params" : {
                    "min_off_time" : 0,
                    "min_on_time" : 12,
                    "number_of_activation_padding": 10,
                    "min_on_power" : 2000
            },
            "methods" : {
                "DeepGRU" : DeepGRU({ 
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/DeepGRU/",
                    "results_folder" : base_path + "results/DeepGRU/",
                    "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                    "plots_folder" : base_path + "plots/DeepGRU/",
                    "appliances" : {
                        "kettle" : {
                            'timewindow' : timestep*sequence_length,
                            'timestep' : timestep,
                            'overlap' :  timestep*sequence_length - timestep,
                            'n_nodes' : 90,
                            'epochs' : epochs,
                            'on_treshold' : 50,
                        }
                    },
                }),
                "ResNet" : ResNet({ 
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/ResNet/",
                    "results_folder" : base_path + "results/ResNet/",
                    "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                    "plots_folder" : base_path + "plots/ResNet/",
                    "appliances" : {
                        "kettle" : {
                            'timewindow' : timestep*sequence_length,
                            'timestep' : timestep,
                            'overlap' :  timestep*sequence_length - timestep,
                            'n_nodes' : 64,
                            'epochs' : epochs,
                            'on_treshold' : 50,
                        }
                    },
                }),
                #"MLP" : MLP({ 
                #    'verbose' : 2,
                #    "training_history_folder" : base_path + "history/MLP/",
                #    "results_folder" : base_path + "results/MLP/",
                #    "checkpoint_folder" : base_path + "temp_weights/MLP/",
                #    "plots_folder" : base_path + "plots/MLP/",
                #    "appliances" : {
                #        "kettle" : {
                #            'timewindow' : timestep*sequence_length,
                #            'timestep' : timestep,
                #            'overlap' :  timestep*sequence_length - timestep,
                #            'n_nodes' : 256,
                #            'epochs' : epochs,
                #            'on_treshold' : 50,
                #            'use_dwt' : True,
                #            'wavelet' : 'db4'
                #        }
                #    },
                #}),
                #"MLP_Raw" : MLP({ 
                #    'model_name' : 'MLP_Raw',
                #    'verbose' : 2,
                #    "training_history_folder" : base_path + "history/MLP_Raw/",
                #    "results_folder" : base_path + "results/MLP_Raw/",
                #    "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                #    "plots_folder" : base_path + "plots/MLP_Raw/",
                #    "appliances" : {
                #        "kettle" : {
                #            'timewindow' : timestep*sequence_length,
                #            'timestep' : timestep,
                #            'overlap' :  timestep*sequence_length - timestep,
                #            'n_nodes' : 256,
                #            'epochs' : epochs,
                #            'on_treshold' : 50,
                #            'use_dwt' : False,
                #        }
                #    },
                #}),
            },
            "model_path" : base_path+"models/",
            "timestep" : 6,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        2 : [(datetime.datetime(2013, 5, 9), datetime.datetime(2013, 10, 10))],
                    }
                },
            },
            "cross_validation" :{
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        2 : [(datetime.datetime(2013, 4, 17), datetime.datetime(2013, 5, 9))],
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        5 : [(datetime.datetime(2014, 6, 30), datetime.datetime(2014, 11, 12))]
                    }
                }
            }
        },
        "washing machine" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'active')],
            "use_activations" : True,
            "appliances_params" : {
                    "min_off_time" : 160,
                    "min_on_time" : 1800,
                    "number_of_activation_padding": 200,
                    "min_on_power" : 20
            },
            "methods" : {
                "DeepGRU" : DeepGRU({ 
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/DeepGRU/",
                    "results_folder" : base_path + "results/DeepGRU/",
                    "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                    "plots_folder" : base_path + "plots/DeepGRU/",
                    "appliances" : {
                        "washing machine" : {
                            'timewindow' : timestep*sequence_length,
                            'timestep' : timestep,
                            'overlap' :  timestep*sequence_length - timestep,
                            'n_nodes' : 90,
                            'epochs' : epochs,
                            'on_treshold' : 50,
                        }
                    },
                }),
                "ResNet" : ResNet({ 
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/ResNet/",
                    "results_folder" : base_path + "results/ResNet/",
                    "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                    "plots_folder" : base_path + "plots/ResNet/",
                    "appliances" : {
                        "washing machine" : {
                            'timewindow' : timestep*sequence_length,
                            'timestep' : timestep,
                            'overlap' :  timestep*sequence_length - timestep,
                            'n_nodes' : 64,
                            'epochs' : epochs,
                            'on_treshold' : 50,
                        }
                    },
                }),
                #"MLP" : MLP({ 
                #    'verbose' : 2,
                #    "training_history_folder" : base_path + "history/MLP/",
                #    "results_folder" : base_path + "results/MLP/",
                #    "checkpoint_folder" : base_path + "temp_weights/MLP/",
                #    "plots_folder" : base_path + "plots/MLP/",
                #    "appliances" : {
                #        "washing machine" : {
                #            'timewindow' : timestep*sequence_length,
                #            'timestep' : timestep,
                #            'overlap' :  timestep*sequence_length - timestep,
                #            'n_nodes' : 256,
                #            'epochs' : epochs,
                #            'on_treshold' : 50,
                #            'use_dwt' : True,
                #            'wavelet' : 'db4'
                #        }
                #    },
                #}),
                #"MLP_Raw" : MLP({ 
                #    'model_name' : 'MLP_Raw',
                #    'verbose' : 2,
                #    "training_history_folder" : base_path + "history/MLP_Raw/",
                #    "results_folder" : base_path + "results/MLP_Raw/",
                #    "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                #    "plots_folder" : base_path + "plots/MLP_Raw/",
                #    "appliances" : {
                #        "washing machine" : {
                #            'timewindow' : timestep*sequence_length,
                #            'timestep' : timestep,
                #            'overlap' :  timestep*sequence_length - timestep,
                #            'n_nodes' : 256,
                #            'epochs' : epochs,
                #            'on_treshold' : 50,
                #            'use_dwt' : False,
                #        }
                #    },
                #}),
            },
            "model_path" : base_path+"models/",
            "timestep" : 6,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        1 : [(datetime.datetime(2013, 3, 18), datetime.datetime(2013, 8, 18))],
                    }
                },
            },
            "cross_validation" :{
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        2 : [(datetime.datetime(2013, 4, 17), datetime.datetime(2013, 10, 10))],
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        5 : [(datetime.datetime(2014, 6, 30), datetime.datetime(2014, 11, 12))]
                    }
                }
            }
        }
    }

    run(experiment)

if __name__ == "__main__":
    run_experiment()
