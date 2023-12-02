from flask import request
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import requests
import datetime


import openai

openai.api_key = os.getenv("OPENAI_API_KEY_IMAGE")


class Crop_Predict(object):
    def __init__(self):
        # self.data = data = pd.read_csv('final_new_crop_data_own_repeat.csv')
        self.data = pd.read_csv("Crop1.csv")
        self.city = pd.read_csv("Ploted_6001.csv")

    def crop(self):
        self.data = shuffle(self.data)

        y = self.data.loc[:, "Crop"]
        labelEncoder_y = LabelEncoder()
        y = labelEncoder_y.fit_transform(y)

        self.data["crop_num"] = y
        X = self.data.loc[:, ["N", "P", "K", "pH", "temp", "climate"]].astype(float)
        y = self.data.loc[:, "crop_num"]

        # Training Model
        from sklearn.neighbors import KNeighborsClassifier

        clf = KNeighborsClassifier(n_jobs=3, n_neighbors=20, weights="distance")
        clf.fit(X, y)

        if request.method == "POST":
            # city_name = request.form["City"]
            N = request.json["N"]
            P = request.json["P"]
            K = request.json["K"]
            pH = request.json["PH"]
            location = request.json["Location"]
            print(location)

            # Use the OpenWeatherMap API to get the weather forecast for the next 15 days
            api_key = os.getenv("OPEN_WEATHER_API_KEY")
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&cnt=15&appid={api_key}"
            response = requests.get(url)
            weather_data = response.json()

            print((float(weather_data["list"][0]["main"]["temp"]) - 273.15))
            Temp = float(weather_data["list"][0]["main"]["temp"]) - 273.15

            # Get the current month
            month = datetime.datetime.now().month

            # Get the current hemisphere
            # You can either hardcode the hemisphere as 'north' or 'south'
            # or you can use the latitude of the location to determine the hemisphere
            hemisphere = "north"

            # Determine the season based on the month and hemisphere
            if (month >= 3 and month <= 6) and hemisphere == "north":
                climate = 1
            elif (month >= 7 and month <= 10) and hemisphere == "north":
                climate = 3
            elif (
                month == 11 or month == 12 or month == 1 or month == 2
            ) and hemisphere == "north":
                climate = 2

            # if (
            #     len(city_name) == 0
            #     and len(N) == 0
            #     and len(P) == 0
            #     and len(K) == 0
            #     and len(pH) == 0
            # ):
            #     return "noData"

            # if len(city_name) != 0:
            #     print(city_name)

            #     npk = self.city[self.city["Location"] == city_name]
            #     for index, row in npk.iterrows():
            #         temp = row["temp"]
            #         climate = row["climate"]
            #         temp1 = temp + 10
            #         temp2 = temp - 7

            #     val = []
            #     for index, row in npk.iterrows():
            #         val = [
            #             row["N"],
            #             row["P"],
            #             row["K"],
            #             row["pH"],
            #             row["temp"],
            #             row["climate"],
            #         ]

            #     columns = ["N", "P", "K", "pH", "temp", "climate"]
            #     values = np.array([val[0], val[1], val[2], val[3], val[4], val[5]])
            #     pred = pd.DataFrame(values.reshape(-1, len(values)), columns=columns)

            #     prediction = clf.predict(pred)
            #     real_pred = clf.predict_proba(pred)
            #     print(real_pred)

            #     lst = []
            #     for i in range(101):
            #         if real_pred[0][i] != 0.0:
            #             lst.append(i)

            #     lt = []
            #     for i in range(20):

            #         load_data = self.data[self.data.index == lst[i]]
            #         for index, row in load_data.iterrows():
            #             if row["temp"] >= temp2 and row["temp"] <= temp1:
            #                 if row["climate"] == climate:
            #                     lt.append(row["Crop"])

            # else:
            print(N, P, K, pH)
            temp = Temp
            temp1 = temp + 10
            temp2 = temp - 7

            columns = ["N", "P", "K", "pH", "temp", "climate"]
            values = np.array([N, P, K, pH, temp, climate])
            pred = pd.DataFrame(values.reshape(-1, len(values)), columns=columns)

            prediction = clf.predict(pred)
            # distances, indices = clf.kneighbors(pred,  n_neighbors=10)
            # prediction
            real_pred = clf.predict_proba(pred)
            print(real_pred)

            lst = []
            for i in range(101):
                if real_pred[0][i] != 0.0:
                    lst.append(i)

            lt = []
            image_list = []
            dict = {}
            for i in range(10):
                load_data = self.data[self.data.index == lst[i]]
                for index, row in load_data.iterrows():
                    if row["temp"] >= temp2 and row["temp"] <= temp1:
                        if row["climate"] == climate:
                            lt.append(row["Crop"])
                            response = openai.Image.create(
                                prompt=row["Crop"],
                                n=1,
                                size="256x256",
                            )
                            image_list.append(response["data"][0]["url"])

            for i in range(len(lt)):
                dict[lt[i]] = image_list[i]

            newList = []
            for i in range(len(dict)):
                newList.append(
                    {"name": list(dict.keys())[i], "img": list(dict.values())[i]}
                )

            print(newList)
            return newList

            return dict
