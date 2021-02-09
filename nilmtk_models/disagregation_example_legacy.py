from nilmtk import DataSet
from  nilmtk.utils import compute_rmse

from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM, Hart85, Mean
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
from six import iteritems
import time



#Loading the datasets
train = DataSet('../../datasets/avEiro_h5/avEiro.h5')
test = DataSet('../../datasets/avEiro_h5/avEiro.h55')

#Dividing the dataset by time window
train.set_window(end="2020-10-30")
test.set_window(start="2020-10-30")

train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec


#For selecting the top X appliances we can use
#top_5_train_elec = train_elec.submeters().select_top_k(k=5)

# A function to test the appliances by desagregating the data.
def predict(clf, test_elec, sample_period, timezone):
    pred = {}
    gt= {}
        
    for i, chunk in enumerate(test_elec.mains().load(physical_quantity = 'power', ac_type = 'apparent', sample_period=sample_period)):
        chunk_drop_na = chunk.dropna()
        pred[i] = clf.disaggregate_chunk(chunk_drop_na)
        gt[i]={}

        for meter in test_elec.submeters().meters:
            # Only use the meters that we trained on (this saves time!)    
            gt[i][meter] = next(meter.load(physical_quantity = 'power', ac_type = 'apparent', sample_period=sample_period))
        gt[i] = pd.DataFrame({k:v.squeeze() for k,v in iteritems(gt[i]) if len(v)}, index=next(iter(gt[i].values())).index).dropna()
        
    # If everything can fit in memory
    gt_overall = pd.concat(gt)
    gt_overall.index = gt_overall.index.droplevel()
    pred_overall = pd.concat(pred)
    pred_overall.index = pred_overall.index.droplevel()

    # Having the same order of columns
    gt_overall = gt_overall[pred_overall.columns]
    
    #Intersection of index
    gt_index_utc = gt_overall.index.tz_convert("UTC")
    pred_index_utc = pred_overall.index.tz_convert("UTC")
    common_index_utc = gt_index_utc.intersection(pred_index_utc)
    
    common_index_local = common_index_utc.tz_convert(timezone)
    gt_overall = gt_overall.loc[common_index_local]
    pred_overall = pred_overall.loc[common_index_local]
    appliance_labels = [m for m in gt_overall.columns.values]
    gt_overall.columns = appliance_labels
    pred_overall.columns = appliance_labels
    return gt_overall, pred_overall

classifiers = {'CO':CombinatorialOptimisation(), 'FHMM':FHMM(), 'Hart85', Hart85(), 'Mean':Mean()}
predictions = {}
sample_period = 1
for clf_name, clf in classifiers.items():
    print("*"*20)
    print(clf_name)
    print("*" *20)
    start = time.time()
    clf.train(train_elec, sample_period=sample_period)
    end = time.time()
    print("Runtime =", end-start, "seconds.")
    gt, predictions[clf_name] = predict(clf, test_elec, sample_period, train.metadata['timezone'])

#Using pretier labels for easier identification
appliance_labels = [m.label() for m in gt.columns.values]
gt.columns = appliance_labels
predictions['CO'].columns = appliance_labels
predictions['FHMM'].columns = appliance_labels


predictions['CO']['Meter 2'].head(300).plot(label="Pred-CO")
predictions['FHMM']['Meter 2'].head(300).plot(label="Pred-HMM")
gt['Meter 2'].head(300).plot(label="GT")
plt.legend()

rmse = {}
for clf_name in classifiers.keys():
    rmse[clf_name] = compute_rmse(gt, predictions[clf_name])

rmse = pd.DataFrame(rmse)
print("*"*20)
print("Results:")
print("*" *20)
print(rmse)