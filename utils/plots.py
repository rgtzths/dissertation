from matplotlib import pyplot as plt
from nilmtk import DataSet

def plot_model_history_classification(model_history, destination_folder):
    
    n_epochs = len(model_history['loss'])

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['loss'], label="Training Loss")
    plt.plot(range(1, n_epochs+1), model_history['val_loss'], label="Validation Loss")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Categorical Cross Entropy")
    plt.title("Evolution of the loss of the model")
    plt.legend()
    fig.savefig(destination_folder+"loss.png", dpi=300.0, bbox_inches='tight', orientation="landscape")

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['accuracy'], label="Training Accuracy")
    plt.plot(range(1, n_epochs+1), model_history['val_accuracy'], label="Validation Accuracy")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Accuracy")
    plt.title("Evolution of the accuracy of the model")
    plt.legend()

    fig.savefig(destination_folder+"acc.png", dpi=300.0, bbox_inches='tight', orientation="landscape")

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['matthews_correlation'], label="Training MCC")
    plt.plot(range(1, n_epochs+1), model_history['val_matthews_correlation'], label="Validation MCC")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Matthews Correlation Coeficient")
    plt.title("Evolution of the MCC of the model")
    plt.legend()
    
    fig.savefig(destination_folder+"mcc.png", dpi=300.0, bbox_inches='tight', orientation="landscape")

    plt.close('all')

def plot_model_history_regression(model_history, destination_folder):
    
    n_epochs = len(model_history['loss'])

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['loss'], label="Training Loss")
    plt.plot(range(1, n_epochs+1), model_history['val_loss'], label="Validation Loss")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("Evolution of the loss of the model")
    plt.legend()
    fig.savefig(destination_folder+"loss.png", dpi=300.0, bbox_inches='tight', orientation="landscape")

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['root_mean_squared_error'], label="Training RMSE")
    plt.plot(range(1, n_epochs+1), model_history['val_root_mean_squared_error'], label="Validation RMSE")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Root Mean Squared Error")
    plt.title("Evolution of the RMSE of the model")
    plt.legend()

    fig.savefig(destination_folder+"rmse.png", dpi=300.0, bbox_inches='tight', orientation="landscape")

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['mean_absolute_error'], label="Training MAE")
    plt.plot(range(1, n_epochs+1), model_history['val_mean_absolute_error'], label="Validation MAE")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Mean Absolute Error")
    plt.title("Evolution of the MAE of the model")
    plt.legend()

    fig.savefig(destination_folder+"mae.png", dpi=300.0, bbox_inches='tight', orientation="landscape")

    plt.close('all')

def compare_errors_classification(model_history_1, model_history_2):

    n_epochs = len(model_history_1['loss'])

    plt.plot(range(1, n_epochs+1), model_history_1['matthews_correlation'], label="Training MCC Model 1")
    plt.plot(range(1, n_epochs+1), model_history_1['val_matthews_correlation'], label="Validation MCC Model 1")

    plt.plot(range(1, n_epochs+1), model_history_2['matthews_correlation'], label="Training MCC Model 2")
    plt.plot(range(1, n_epochs+1), model_history_2['val_matthews_correlation'], label="Validation MCC Model 2")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Matthews Correlation Coeficient")
    plt.title("Comparison of Matthews Correlation Coeficient")
    plt.legend()

    plt.show()

def compare_errors_regfression(model_history_1, model_history_2):

    n_epochs = len(model_history_1['loss'])

    plt.plot(range(1, n_epochs+1), model_history_1['mean_absolute_error'], label="Training MAE Model 1")
    plt.plot(range(1, n_epochs+1), model_history_1['val_mean_absolute_error'], label="Validation MAE Model 1")

    plt.plot(range(1, n_epochs+1), model_history_2['mean_absolute_error'], label="Training MAE Model 2")
    plt.plot(range(1, n_epochs+1), model_history_2['val_mean_absolute_error'], label="Validation MAE Model 2")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Mean Absolute Error")
    plt.title("Comparisson of Mean Absolute Error")
    plt.legend()

    plt.show()


def plot_predictions(model, dataset_path, app_name, building, beginning, end ):
    
    dataset = DataSet(dataset_path)
    dataset.set_window(start=beginning, end=end)

    app_readings = next(dataset.buildings[building].elec[app_name].load())
    main_readings = next(dataset.buildings[building].elec[app_name].load())

    predictions = model.disaggregate_chunk([app_readings])

    plt.plot(main_readings.index, main_readings["power"]["apparent"], label=app_name)
    plt.plot(app_readings.index, app_readings["power"]["apparent"], label="Aggregated")
    plt.plot(app_readings.index, predictions[0], label="Predicted " + app_name)

    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.title("Aggregated Readings, " + app_name + " readings and predicted readings")
    plt.legend()

    plt.show()



''' Experiment Example
experiment = {
    "fridge" : {
        "location" : '../../datasets/ukdale/ukdale.h5',
        "houses" : {
            1 : [
                ("2014-02-20", "2014-02-28"),
                ("2016-09-10", "2016-09-17")
            ],
            2 : [
                ("2013-05-23", "2013-05-30"),
                ("2013-07-10", "2013-07-17")
            ],
        }
        
    },
    "microwave" : {
        "location" : '../../datasets/ukdale/ukdale.h5',
        "houses" : {
            1 : [
                ("2014-02-20", "2014-02-28"),
                ("2016-09-10", "2016-09-17")
            ],
        }  
    },
}
'''
def plot_experiment(experiment):
    for app in experiment:
        dataset = DataSet(experiment[app]["location"]) 

        for house in experiment[app]["houses"]:

            for timeperiod in experiment[app]["houses"][house]:
                
                dataset.set_window(start=timeperiod[0],end=timeperiod[1])

                mains = next(dataset.buildings[house].elec.mains().load())

                app_df = next(dataset.buildings[house].elec[app].load())


                plt.plot(mains.index, mains["power"]["apparent"], label="Mains Energy")
                plt.plot(app_df.index, app_df["power"]["active"], label=app + " Energy")
                plt.xlabel("Time")
                plt.ylabel("Power")
                plt.title("Mains and " + app + " energy consumption from house" + str(house))
                plt.legend()

                plt.show()

def plot_signature(dataset_location, building, appliance, beggining, end):
    dataset = DataSet(dataset_location)

    dataset.set_window(start=beggining,end=end)

    mains = next(dataset.buildings[building].elec.mains().load())
    app_df = next(dataset.buildings[building].elec[appliance].load())
    plt.plot(mains.index, mains["power"]["apparent"], label="Mains Energy")
    plt.plot(app_df.index, app_df["power"]["active"], label=appliance + " Energy")


    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.title("Mains and " + appliance + " energy consumption from house" + str(building))
    plt.legend()

    plt.show()