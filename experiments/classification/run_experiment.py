
import sys
sys.path.insert(1, "../../utils")
from utils import load_data, load_data_h5

def run(experiment):
    results = {}
    for app in experiment:
        X_train = []
        y_train = []

        X_cv = []
        y_cv = []

        X_test = []
        y_test = []

        results[app] = {}

        print("...................Loading Train Data for %s..................." % (app))
        for dataset in experiment[app]["train"]:
            if experiment[app]["train"][dataset]["location"].split(".")[-1] != "h5":
                x, y = load_data(
                    experiment[app]["train"][dataset]["location"],
                    app, 
                    experiment[app]["train"][dataset]["houses"],
                    experiment[app]["timestep"],
                    experiment[app]['mains_columns'],
                    experiment[app]['appliance_columns']
                )
            else:
                x, y = load_data_h5(
                    experiment[app]["train"][dataset]["location"],
                    app, 
                    experiment[app]["train"][dataset]["houses"],
                    experiment[app]["timestep"],
                    experiment[app]['mains_columns'],
                    experiment[app]['appliance_columns'],
                    experiment[app]["use_activations"],
                    experiment[app].get("appliances_params", None)
                )
                x_cv, y_cross = load_data_h5(
                    experiment[app]["cross_validation"][dataset]["location"],
                    app, 
                    experiment[app]["cross_validation"][dataset]["houses"],
                    experiment[app]["timestep"],
                    experiment[app]['mains_columns'],
                    experiment[app]['appliance_columns'],
                    experiment[app]["use_activations"],
                    experiment[app].get("appliances_params", None)
                )
                X_cv += x_cv
                y_cv += y_cross

            X_train += x
            y_train += y
        
        train_data = { 
            app : {
                "mains" : X_train,
                "appliance" : y_train
            }
        }
        if len(X_cv) >= 1:
            cv_data = {
                app : {
                    "mains" : X_cv,
                    "appliance" : y_cv
                }
            }
        else:
            cv_data = None
        for method in experiment[app]["methods"]:
            print("\n\nTraining %s" % (method))
            experiment[app]["methods"][method].partial_fit(train_data, cv_data)

            #experiment[app]["methods"][method].save_model(experiment[app]["model_path"] + method.lower())

        print("\n\n...................Loading Test Data for %s..................." % (app))
        for dataset in experiment[app]["test"]:
            if experiment[app]["test"][dataset]["location"].split(".")[-1] != "h5":
                x, y = load_data(
                            experiment[app]["test"][dataset]["location"],
                            app, 
                            experiment[app]["test"][dataset]["houses"],
                            experiment[app]["timestep"],
                            experiment[app]['mains_columns'],
                            experiment[app]['appliance_columns']
                )
            else:
                x, y = load_data_h5(
                    experiment[app]["test"][dataset]["location"],
                    app, 
                    experiment[app]["test"][dataset]["houses"],
                    experiment[app]["timestep"],
                    experiment[app]['mains_columns'],
                    experiment[app]['appliance_columns'],
                    experiment[app]["use_activations"],
                    experiment[app].get("appliances_params", None)
                )
            
            X_test += x
            y_test += y
                
        for method in experiment[app]["methods"]:
            print("\n\n...................Testing %s..................." % (method))
            res = experiment[app]["methods"][method].disaggregate_chunk(X_test, [(app, y_test)])

            results[app][method] = res

    for app in experiment:
        print("\n\n...................Results Obtained for %s..................." % (app))
        for method in results[app]:
            print("%10s" % (method), end="")
            print("%10.2f" % (results[app][method][app]), end="")
            print()
        print()
