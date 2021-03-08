import dataset_loader

def run(experiment):
    results = {}
    for app in experiment:
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        results[app] = {}

        print("Loading Train Data for %s" % (app))
        for dataset in experiment[app]["train"]:
            x, y = dataset_loader.load_data(
                        experiment[app]["train"][dataset]["location"],
                        app, 
                        experiment[app]["train"][dataset]["houses"]
                    )
            for i in range(0, len(x)):
                X_train.append(x[i])
                y_train.append(y[i])
        print("Loading Test Data for %s" % (app))
        for dataset in experiment[app]["test"]:
            x, y = dataset_loader.load_data(
                        experiment[app]["test"][dataset]["location"],
                        app, 
                        experiment[app]["test"][dataset]["houses"]
            )
            for i in range(0, len(x)):
                X_test.append(x[i])
                y_test.append(y[i])
    
        for method in experiment[app]["methods"]:
            print("Training %s" % (method))
            experiment[app]["methods"][method].partial_fit(X_train, y_train, app)

            experiment[app]["methods"][method].save_model(experiment[app]["model_path"] + method.split("_")[0].lower())

            print("Testing %s" % (method))
            res = experiment[app]["methods"][method].disaggregate_chunk(X_test, y_test, app)

            results[app][method] = res
    for app in experiment:
        print("Results Obtained for %s" % (app))
        for res in results[app]:
            print("%20s" % (res), end="\t")
        print()
        for res in results[app]:
            print("%20.2f" % (results[app][res]), end="\t")
        print()
