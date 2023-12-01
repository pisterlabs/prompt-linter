from .SOAIDataHandler import SOAIDataHandler

import pandas as pd
import glob
import logging
import json
import os

logger = logging.getLogger()


## Class which handles the loading of data from disk.
class SOAIDiskHandler(SOAIDataHandler):

    ## Constructor
    def __init__(self):
        pass

    ## Load data from OpenAir Cologne
    #
    # @param path Path to folder with files
    # @param selectValidData Boolean if only valid data shall be loaded
    # @returns Pandas data frame with measurment values depending on time
    def fGetOpenAir(self, path=os.environ.get("SOAI") + "/data/openair/", selectValidData=True):
        logger.debug(f"Load OpenAir Cologne data from {path}.")

        # Since the measurments are saved in multiple files, use a glob-string and wildcards in order to load the data
        filesOpenAir = glob.glob(path + "*.parquet")
        listOpenAir = []
        for filename in filesOpenAir:
            logger.debug(f"\t- Load {filename}")
            df = pd.read_parquet(filename)
            listOpenAir.append(df)

        # Sort the values depending on time
        dataOpenAir = pd.concat(listOpenAir, sort=False)

        # Check the OpenAir data
        dataOpenAir = self._fCheckOpenAir(dataOpenAir, selectValidData)
        dataOpenAir = dataOpenAir.set_index("timestamp", drop=True)

        return dataOpenAir

    ## Load properties of OpenAir Cologn sensors
    #
    # @param pathToFile Path to file where sensor data is saved
    # @param selectValidData Boolean if only valid sensor data shall be loaded
    # @returns Pandas data frame with location of the sensors
    def fGetOpenAirSensors(self, pathToFile=os.environ.get("SOAI") + "/data/openair/sensors.json"):
        logger.debug(f"Load open air sensor properties from {pathToFile}.")

        # Open json data where properties of senesors are encoded
        with open(pathToFile) as f:
            data = json.load(f)

        # Create a data frame with location information about the sensors
        df = pd.DataFrame(columns=["sensorID", "lon", "lat"])
        for feed in data["features"]:
            lonCoordinate = feed["geometry"]["coordinates"][0]
            latCoordinate = feed["geometry"]["coordinates"][1]
            mqtt_id = feed["properties"]["mqtt_id"].split("-")[0]

            df = df.append({"sensorID": mqtt_id, "lon": lonCoordinate, "lat": latCoordinate}, ignore_index=True)

        return df

    ## Load data from Luftdaten.info
    #
    # @param path Path to folder with files
    # @param selectValidData Boolean if only valid data shall be loaded
    # @returns Pandas data frame with measurment values depending on time
    def fGetLanuv(self, path=os.environ.get("SOAI") + "/data/lanuv/", selectValidData=False):
        logger.debug(f"Load Lanuv data from {path}.")

        # Since the measurments are saved in multiple files, use a glob-string and wildcards in order to load the data
        filesLanuv = glob.glob(path + "*.parquet")
        listLanuv = []
        for filename in filesLanuv:
            logger.debug(f"\t- Load {filename}")
            df = pd.read_parquet(filename)
            listLanuv.append(df)

        # Concat data
        dataLanuv = pd.concat(listLanuv, ignore_index=True, sort=False)

        # Check the Lanuv data
        dataLanuv = self._fCheckLanuv(dataLanuv, selectValidData)
        dataLanuv = dataLanuv.set_index("timestamp", drop=True)

        return dataLanuv

    ## Load properties of OpenAir Cologn sensors
    #
    # @param pathToFile Path to file where sensor data is saved
    # @returns Pandas data frame with location of the sensors
    def fGetLanuvSensors(self, pathToFile=os.environ.get("SOAI") + "/data/lanuv/sensors.csv"):
        logger.debug(f"Load lanuv sensor properties from {pathToFile}.")

        df = pd.read_csv(pathToFile, sep=";")

        return df

    ## Load the traffic data from the available files
    def fGetTrafficData(self, path=os.environ.get("SOAI") + "/data/traffic/", selectValidData=False, pixelSize=None):
        logger.debug(f"Load traffic data from {path}.")

        # Since the data are saved in multiple files, use a glob-string and wildcards in order to load the data
        files = glob.glob(path + "*.csv")
        listData = []
        for filename in files:
            infos = filename[filename.rfind("/") + 1:].split("_")
            logger.debug(f"\t- Load {filename} with informations {infos}.")

            if len(infos) < 3:
                logger.error("csv-file with traffic data have a different format as epected. Expect filename with format station_pixel_time.csv")
                raise Exception("csv-file with traffic data have a different format as epected.")
            if pixelSize is not None and pixelSize != int(infos[1]):
                logger.debug(f"Skip file since pixel size is set to {pixelSize}.")
                continue

            df = pd.read_csv(filename, sep=",")

            if len(df) == 0:
                logger.warning("Not data was found. Skip this file.")
                continue

            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date", drop=True)
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
            df.index = df.index.tz_localize(None)
            df.loc[df['green_X'].isnull() == False, 'green'] = 1
            df.loc[df['orange_X'].isnull() == False, 'orange'] = 1
            df.loc[df['red_X'].isnull() == False, 'red'] = 1
            df.loc[df['brown_X'].isnull() == False, 'brown'] = 1
            df = df.resample('15min').agg({"green": 'sum', "orange": 'sum', "red": "sum", "brown": "sum"})
            df["pixelCounterSensor"] = df["green"] + df["orange"] + df["red"] + df["brown"]
            # df = df[df["pixelCounterSensor"] > 0]

            df["rgreen"] = df.apply(lambda x: x["green"] / x["pixelCounterSensor"] if x["pixelCounterSensor"] > 0 else 0, axis=1)
            df["rorange"] = df.apply(lambda x: x["orange"] / x["pixelCounterSensor"] if x["pixelCounterSensor"] > 0 else 0, axis=1)
            df["rred"] = df.apply(lambda x: x["red"] / x["pixelCounterSensor"] if x["pixelCounterSensor"] > 0 else 0, axis=1)
            df["rbrown"] = df.apply(lambda x: x["brown"] / x["pixelCounterSensor"] if x["pixelCounterSensor"] > 0 else 0, axis=1)

            df["sensorID"] = [infos[0] for i in range(len(df))]
            df["pixel"] = [int(infos[1]) for i in range(len(df))]

            df = df.reset_index()

            listData.append(df)

        # Concat data
        try:
            data = pd.concat(listData, ignore_index=True, sort=False)
        except Exception as e:
            logger.error(f"{e}")
            logger.error("No data available? Correct path?")
            return pd.DataFrame()

        # Check the traffic data
        data = self._fCheckTraffic(data, selectValidData)
        data = data.set_index("date", drop=True)

        return data
