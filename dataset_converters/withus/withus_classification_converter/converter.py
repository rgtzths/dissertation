from os.path import join, isdir, isfile
from os import listdir
import re
import json
import pandas as pd
import numpy as np
import requests
from io import StringIO
import datetime

from nilmtk.measurement import LEVEL_NAMES

import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../../../utils")
import utils

from data_clean import clean_data


def download_and_convert(base_url, username, password, column_mapping, appliances_mapping, timestep, interpolate, output_path):

    response = requests.get(base_url + 'households', auth=(username,password)).json()

    houses = [ house["household"] for house in response if house["household"] != "placeholder_house"]

    week_td = pd.Timedelta(7, unit="D")

    for house in houses:
        
        response = requests.get(base_url + 'household_devices/'+house, auth=(username,password)).json()

        load_data = 0
        for app in response:
            if app["device_type"] != "meter":
                if "description" in app["metadata"] and app["metadata"]["description"] in appliances_mapping:
                    load_data = 1
                    break
        if load_data:
            for app in response:
                dfs = []
                if app["device_type"] != "meter":
                    if "description" in app["metadata"] and app["metadata"]["description"] in appliances_mapping:
                        print("Loading appliance data ("+app["metadata"]["description"]+ ") from house " + house)

                        dates =  requests.get(base_url + 'device/'+app["device"]+'/interval' , auth=(username,password)).json()
                        
                        begining_date = datetime.datetime.strptime(dates["startDateStr"], "%Y-%m-%dT%H:%M:%S.%fZ")
                        end_date = datetime.datetime.strptime(dates["endDateStr"], "%Y-%m-%dT%H:%M:%S.%fZ")

                        if (end_date - begining_date).days > 7 :
                            begining = begining_date

                            end = begining_date + week_td

                            while(end < end_date):

                                q_beginning = begining.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"
                                q_end = end.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"

                                response =  StringIO (requests.get(base_url + 'plug/individual?device_id='+app["device"]+"&begin="+q_beginning+"&end="+q_end , 
                                                    auth=(username,password), 
                                                    headers={'Content-Type': 'text/csv'}).content.decode("utf-8") )

                                df = pd.read_csv(response, sep=",")

                                del df['plug_id']
                                del df['gateway_id']
                                del df['device_type']
                                del df['period']

                                dfs.append(df)
                                begining = end
                                end += week_td
                            
                            end = end_date

                            q_beginning = begining.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"
                            q_end = end.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"

                            response =  StringIO (requests.get(base_url + 'plug/individual?device_id='+app["device"]+"&begin="+q_beginning+"&end="+q_end , 
                                                auth=(username,password), 
                                                headers={'Content-Type': 'text/csv'}).content.decode("utf-8") )

                            df = pd.read_csv(response, sep=",")

                            del df['plug_id']
                            del df['gateway_id']
                            del df['device_type']
                            del df['period']

                            dfs.append(df)

                else:
                    print("Loading aggregated data from house " + house)
                    dates =  requests.get(base_url + 'device/'+app["device"]+'/interval' , auth=(username,password)).json()
                    
                    begining_date = datetime.datetime.strptime(dates["startDateStr"], "%Y-%m-%dT%H:%M:%S.%fZ")
                    end_date = datetime.datetime.strptime(dates["endDateStr"], "%Y-%m-%dT%H:%M:%S.%fZ")

                    if (end_date - begining_date).days > 7 :
                        begining = begining_date

                        end = begining_date + week_td

                        while(end < end_date):

                            q_beginning = begining.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"
                            q_end = end.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"

                            response = StringIO(
                                        requests.get(base_url + 'mono-meter/individual?device_id='+app["device"]+"&begin="+q_beginning+"&end="+q_end , 
                                                    auth=(username,password),
                                                    headers={'Content-Type': 'text/csv'} 
                                                    ).content.decode("utf-8") 
                                        )

                            df = pd.read_csv(response, sep=",")

                            del df['meter_id']
                            del df['gateway_id']
                            del df['device_type']
                            del df['period']

                            dfs.append(df)
                            begining = end
                            end += week_td
                        
                        end = end_date

                        q_beginning = begining.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"
                        q_end = end.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"

                        response = StringIO(
                                        requests.get(base_url + 'mono-meter/individual?device_id='+app["device"]+"&begin="+q_beginning+"&end="+q_end , 
                                                    auth=(username,password),
                                                    headers={'Content-Type': 'text/csv'} 
                                                    ).content.decode("utf-8") 
                                        )

                        df = pd.read_csv(response, sep=",")

                        del df['meter_id']
                        del df['gateway_id']
                        del df['device_type']
                        del df['period']

                        dfs.append(df)

                if len(dfs) > 0:

                    data = pd.concat(dfs, axis=0)
#
                    #data = clean_data(data, timestep, interpolate)
#
                    #data.columns = pd.MultiIndex.from_tuples([column_mapping[c] for c in data.columns.values])
                    #data.columns.set_names(LEVEL_NAMES, inplace=True)
#
                    #utils.create_path('{}/house_{}'.format(output_path, house))
                    #df.to_csv('{}/house_{}/{}.csv'.format(output_path, house, appliance))
                    #print(data)
        


username = 'ml_login'
password = 'ml_pw_enc'

##Change
column_mapping = {
    "power" : ("power", "apparent"),
    "vrms" : ("voltage", "")
}

appliances_mapping = {
    "Fridge" : "fridge", 
    "Dishwasher": "dish washer", 
    "Laundry machine" : "washing machine", 
    "Heat pump" : "heat pump"
}

timestep = 2

interpolate = "previous"

base_url = "http://withus.av.it.pt/api/v1/ed/"

output_path = "../../../../datasets/withus_classification"

download_and_convert(base_url, username, password, column_mapping, appliances_mapping, timestep, interpolate, output_path)