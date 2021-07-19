
import pandas as pd
import requests
from io import StringIO
import datetime

from nilmtk.measurement import LEVEL_NAMES

import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../../../utils")
import utils

from data_clean import clean_data

def load_period(base_url, device, beginning, end, username, password, is_meter):

    if is_meter:
        response = StringIO(
                    requests.get(base_url + 'mono-meter/individual?device_id='+device+"&begin="+beginning+"&end="+end , 
                                auth=(username,password),
                                headers={'Content-Type': 'text/csv'},
                                timeout=None 
                                ).content.decode("utf-8") 
                    )
    else:
        response =  StringIO (
                    requests.get(base_url + 'plug/individual?device_id='+device+"&begin="+beginning+"&end="+end , 
                                auth=(username,password), 
                                headers={'Content-Type': 'text/csv'}, 
                                timeout=None
                                ).content.decode("utf-8") 
                    )

    df = pd.read_csv(response, sep=",")

    if is_meter:
        del df['meter_id']
    else: 
        del df['plug_id']
    del df['gateway_id']
    del df['device_type']
    del df['period']
    
    df.index = pd.to_datetime(df["timestamp"], unit='ms')
    df.index = df.index.round("s", ambiguous=False)

    del df["timestamp"]
    
    #Sort index and drop duplicates
    df = df.sort_index()
    dups_in_index = df.index.duplicated(keep='first')
    if dups_in_index.any():
        df = df[~dups_in_index]

    return df


def download_and_convert(base_url, username, password, appliances_mapping, timestep, interpolate, output_path):

    response = requests.get(base_url + 'households', auth=(username,password), timeout=None).json()

    houses = [ house["household"] for house in response if house["household"] != "placeholder_house" and house["household"] not in ["006056131261"]]

    week_td = pd.Timedelta(7, unit="D")

    for house in houses:
        
        response = requests.get(base_url + 'household_devices/'+house, auth=(username,password), timeout=None).json()

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
                        appliance = appliances_mapping[app["metadata"]["description"]]

                        dates =  requests.get(base_url + 'device/'+app["device"]+'/interval' , auth=(username,password), timeout=None).json()
                        
                        begining_date = datetime.datetime.strptime(dates["startDateStr"], "%Y-%m-%dT%H:%M:%S.%fZ")
                        end_date = datetime.datetime.strptime(dates["endDateStr"], "%Y-%m-%dT%H:%M:%S.%fZ")

                        if (end_date - begining_date).days > 7 :
                            begining = begining_date

                            end = begining_date + week_td

                            while(end < end_date):

                                q_beginning = begining.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"
                                q_end = end.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"

                                df = load_period(base_url, app["device"], q_beginning, q_end, username, password, False)

                                dfs.append(df)
                                begining = end
                                end += week_td
                            
                            end = end_date

                            q_beginning = begining.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"
                            q_end = end.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"

                            df = load_period(base_url, app["device"], q_beginning, q_end, username, password, False)

                            dfs.append(df)

                else:
                    print("Loading aggregated data from house " + house)
                    appliance = "mains"
                    print(requests.get(base_url + 'device/'+app["device"]+'/interval' , auth=(username,password), timeout=None).text)
                    dates =  requests.get(base_url + 'device/'+app["device"]+'/interval' , auth=(username,password), timeout=None).json()
                    begining_date = datetime.datetime.strptime(dates["startDateStr"], "%Y-%m-%dT%H:%M:%S.%fZ")
                    end_date = datetime.datetime.strptime(dates["endDateStr"], "%Y-%m-%dT%H:%M:%S.%fZ")

                    if (end_date - begining_date).days > 7 :
                        begining = begining_date

                        end = begining_date + week_td

                        while(end < end_date):

                            q_beginning = begining.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"
                            q_end = end.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"

                            df = load_period(base_url, app["device"], q_beginning, q_end, username, password, True)
                            dfs.append(df)
                            begining = end
                            end += week_td
                        
                        end = end_date

                        q_beginning = begining.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"
                        q_end = end.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3] + "Z"

                        df = load_period(base_url, app["device"], q_beginning, q_end, username, password, True)

                        dfs.append(df)

                if len(dfs) > 0:

                    data = pd.concat(dfs, axis=0, sort=True)
                    dups_in_index = data.index.duplicated(keep='first')
                    if dups_in_index.any():
                        print("Duplicates")
                        data = data[~dups_in_index]
                    
                    data = clean_data(data, timestep, interpolate)

                    utils.create_path('{}/house_{}'.format(output_path, house))
                    data.to_csv('{}/house_{}/{}.csv'.format(output_path, house, appliance))

username = 'ml_login'
password = 'ml_pw_enc'

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

download_and_convert(base_url, username, password, appliances_mapping, timestep, interpolate, output_path)