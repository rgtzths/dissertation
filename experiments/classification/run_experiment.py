
import sys
sys.path.insert(1, "../../feature_extractors")
import dataset_loader

def run(experiment):
    results = {}
    for app in experiment:
        X_train = []
        y_train = []

        X_test = []
        y_test = []

        results[app] = {}

        print("...................Loading Train Data for %s..................." % (app))
        for dataset in experiment[app]["train"]:
            x, y = dataset_loader.load_data(
                        experiment[app]["train"][dataset]["location"],
                        app, 
                        experiment[app]["train"][dataset]["houses"],
                        experiment[app]["timestep"],
                        experiment[app]['mains_columns'],
                        experiment[app]['appliance_columns']
                    )
            for i in range(0, len(x)):
                X_train.append(x[i])
                y_train.append(y[i])
            
        for method in experiment[app]["methods"]:
            print("\n\nTraining %s" % (method))
            experiment[app]["methods"][method].partial_fit(X_train, [(app, y_train)])

            #experiment[app]["methods"][method].save_model(experiment[app]["model_path"] + method.lower())

        print("\n\n...................Loading Test Data for %s..................." % (app))
        for dataset in experiment[app]["test"]:
            x, y = dataset_loader.load_data(
                        experiment[app]["test"][dataset]["location"],
                        app, 
                        experiment[app]["test"][dataset]["houses"],
                        experiment[app]["timestep"],
                        experiment[app]['mains_columns'],
                        experiment[app]['appliance_columns']
            )
            for i in range(0, len(x)):
                X_test.append(x[i])
                y_test.append(y[i])
                
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
