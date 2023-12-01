import streamlit as st
import time
import cv2
import dlib
import pyautogui
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import fitbit
import matplotlib.dates as mdates
import arrow  # Arrow is a really useful date time helper library
from openai import OpenAI
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import firebase_admin
from firebase_admin import storage
from PIL import Image

pyautogui.FAILSAFE = False

def get_bounding_box(landmarks):
    # Find min and max landmarks based on X coordinate, and then select the X coordinate
    min_x = min(landmarks, key=lambda p: p.x).x
    max_x = max(landmarks, key=lambda p: p.x).x
    # Find min and max landmarks based on Y coordinate, and then select the Y coordinate
    min_y = min(landmarks, key=lambda p: p.y).y
    max_y = max(landmarks, key=lambda p: p.y).y

    return min_x, max_x, min_y, max_y


def filter_for_iris(eye_image):
    # Convert to grayscale
    eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)

    # Blur frame
    eye_image = cv2.bilateralFilter(eye_image, 10, 15, 15)

    # Adjust brightness
    eye_image = cv2.equalizeHist(eye_image)

    # Find dark parts of frame
    iris_image = 255 - \
        cv2.threshold(eye_image, 50, 255, cv2.THRESH_BINARY)[1]

    return iris_image


def find_iris_location(iris_image):
    # Find contours
    contours, _ = cv2.findContours(
        iris_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort in ascending order by area
    contours = sorted(contours, key=cv2.contourArea)

    try:
        # Find center of largest contour
        moments = cv2.moments(contours[-1])
        # m[i, j] = Sum(x ^ i * y ^ j * brightness at (x, y))
        x = int(moments['m10'] / moments['m00'])
        y = int(moments['m01'] / moments['m00'])
    except (IndexError, ZeroDivisionError):
        # Assume there is no iris
        return None

    return x, y


def crop(image, bbox):
    return image[bbox[2]:bbox[3], bbox[0]:bbox[1]]

def switch_analyze():
    st.session_state.page = "analyze"

def page_track():

    st.write("Productivity Tracking")
    st.button("End Tracking", key="one", on_click=switch_analyze)

    # Open a CSV file for writing eye gaze coordinates
    csv_file = open('eye_gaze_coordinates.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'Normalized X', 'Normalized Y', 'On Screen', 'Awake', 'Blinking', 'Blinks', 'Score'])


    find_faces = dlib.get_frontal_face_detector()
    find_landmarks = dlib.shape_predictor(
    './shape_predictor_68_face_landmarks.dat')

    cap = cv2.VideoCapture(0)

    top_left_average_offset = None
    bottom_right_average_offset = None
    last_time_i_told_the_user_to_look_somewhere = None
    left_eye_width = None
    left_eye_height = None
    right_eye_height = None
    right_eye_width = None
    blinks = [0 for i in range(30)]




    while True:

        if cv2.waitKey(1) & 0xFF == ord('q'):
            csv_file.close()
            break

        _, frame = cap.read()

        for face_bounding_box in find_faces(frame):
            landmarks = find_landmarks(frame, face_bounding_box).parts()

            left_bbox = get_bounding_box(landmarks[36:42])
            right_bbox = get_bounding_box(landmarks[42:48])

            left_eye_frame = crop(frame, left_bbox)
            right_eye_frame = crop(frame, right_bbox)

            left_iris = filter_for_iris(left_eye_frame)
            right_iris = filter_for_iris(right_eye_frame)

            left_iris_location = find_iris_location(left_iris)
            right_iris_location = find_iris_location(right_iris)

            left_eye_center = (
                (landmarks[36].x + landmarks[39].x) // 2 -
                left_bbox[0]), ((landmarks[36].y + landmarks[39].y) // 2 - left_bbox[2])

            right_eye_center = (
                (landmarks[42].x + landmarks[45].x) // 2 -
                right_bbox[0]), ((landmarks[42].y + landmarks[45].y) // 2 - right_bbox[2])

            curr_left_eye_width = left_bbox[1] - left_bbox[0]
            curr_left_eye_height = left_bbox[3] - left_bbox[2]
            curr_right_eye_width = right_bbox[1] - right_bbox[0]
            curr_right_eye_height = right_bbox[3] - right_bbox[2]
            awake = False
            blinking = False
            onScreen = False

            left_iris_offset = None
            right_iris_offset = None

            if left_iris_location is not None:
                left_iris_offset = (
                    left_iris_location[0] - left_eye_center[0], left_iris_location[1] - left_eye_center[1])

                cv2.circle(left_eye_frame, left_iris_location, 2, (0, 0, 255), -1)

            if right_iris_location is not None:
                right_iris_offset = (
                    right_iris_location[0] - right_eye_center[0], right_iris_location[1] - right_eye_center[1])

                cv2.circle(right_eye_frame, right_iris_location,
                        2, (0, 0, 255), -1)

            if left_iris_offset is not None and right_iris_offset is not None:
                average_offset = (left_iris_offset[0] + right_iris_offset[0]) // 2,\
                    (left_iris_offset[1] + right_iris_offset[1]) // 2

                needs_calibration = (top_left_average_offset is None) or (
                    bottom_right_average_offset is None) or (left_eye_width is None) or (right_eye_width is None) or (left_eye_height is None) or (right_eye_height is None)

                if needs_calibration:
                    if last_time_i_told_the_user_to_look_somewhere is None:
                        if top_left_average_offset is None:
                            print("Look at top left corner")
                        elif bottom_right_average_offset is None:
                            print("Look at bottom right corner")
                        else:
                            print("Look at center")

                        last_time_i_told_the_user_to_look_somewhere = time.time()
                    elif time.time() >= last_time_i_told_the_user_to_look_somewhere + 5:
                        if top_left_average_offset is None:
                            top_left_average_offset = average_offset
                        elif bottom_right_average_offset is None:
                            bottom_right_average_offset = average_offset
                        elif left_eye_width is None:
                            # print("HERE")
                            left_eye_width = curr_left_eye_width
                            left_eye_height = curr_left_eye_height
                            right_eye_width = curr_right_eye_width
                            right_eye_height = curr_right_eye_height

                        last_time_i_told_the_user_to_look_somewhere = None
                else:
                    min_x, min_y = top_left_average_offset
                    max_x, max_y = bottom_right_average_offset
                    # print(left_eye_width, left_eye_height, right_eye_width, right_eye_height, curr_left_eye_width, curr_left_eye_height, curr_right_eye_width, curr_right_eye_height)
                    if left_eye_height is not None and right_eye_height is not None and left_eye_width is not None and right_eye_width is not None and curr_left_eye_height is not None and curr_right_eye_height is not None and curr_left_eye_width is not None and curr_right_eye_width is not None:
                        if (left_eye_height)/(left_eye_width)*0.8 > (curr_left_eye_height)/(curr_left_eye_width) and (right_eye_height)/(right_eye_width)*0.8 > (curr_right_eye_height)/(curr_right_eye_width):
                            # print("WAKE UP!")
                            blinks[int(time.time()) % 30] = 0
                        else:
                            blinks[int(time.time()) % 30] = 1
                            # print("YOU'RE AWAKE GOOD JOB!")
                        print(blinks)
                        if(sum(blinks) > 20):
                            print("WAKE UP!")
                            awake = True
                        else:
                            print("YOU'RE AWAKE GOOD JOB!")
                            awake = False
                        if(sum(blinks) < 6):
                            print("REST YOUR EYES!")
                            blinking = True
                        else:
                            print("YOU'RE RESTED GOOD JOB!")
                            blinking = False
                        normalized_x = 1920 * (average_offset[0] - min_x) / (max_x - min_x)
                        normalized_y = 1080 * (average_offset[1] - min_y) / (max_y - min_y)
                        if normalized_x < 0 or normalized_x > 1920 or normalized_y < 0 or normalized_y > 1080:
                            print("User is looking off-screen!")
                            onScreen = False
                        else:
                            print("User is looking at the screen!")
                            onScreen = True
                        # Write eye gaze coordinates to the CSV file
                        timestamp = time.time()
                        csv_writer.writerow([timestamp, normalized_x, normalized_y, onScreen, awake, blinking])

                    # pyautogui.moveTo(
                    #     1920 * (average_offset[0] - min_x) / (max_x - min_x), 1080 * (average_offset[1] - min_y) / (max_y - min_y))

        cv2.imshow('frame', frame)

def display_heatmap(csv_path, title):
    # Load CSV data
    df = pd.read_csv(csv_path)

    # Extract relevant columns
    scatter_data = df[['Normalized X', 'Normalized Y']]

    # Calculate kernel density estimate for the data
    kde = gaussian_kde(scatter_data.T)

    # Create a grid of coordinates for the heatmap
    x_grid, y_grid = np.mgrid[scatter_data['Normalized X'].min():scatter_data['Normalized X'].max():100j,
                                scatter_data['Normalized Y'].max():scatter_data['Normalized Y'].min():100j]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = np.reshape(kde(positions).T, x_grid.shape)

    # Create a heatmap without scatter plot
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(z), cmap=plt.cm.viridis, extent=[scatter_data['Normalized X'].min(),
                                                    scatter_data['Normalized X'].max(),
                                                    scatter_data['Normalized Y'].min(),
                                                    scatter_data['Normalized Y'].max()],
            alpha=0.5, aspect='auto')
    ax.set_xlabel('Normalized X')
    ax.set_ylabel('Normalized Y')
    ax.set_title(title)

    # Display the heatmap using Streamlit
    st.pyplot(fig)

def page_analyze():
    st.header("Productivity Analytics")
    if st.button("Start Tracking"):
        st.session_state.page = "track"
    
    # Load CSV data
    df = pd.read_csv('eye_gaze_coordinates.csv')

    # Extract relevant columns
    scatter_data = df[['Normalized X', 'Normalized Y']]

    # Create a scatter plot using Matplotlib
    fig, ax = plt.subplots()
    ax.scatter(scatter_data['Normalized X'], scatter_data['Normalized Y'])
    ax.set_xlabel('Normalized X')
    ax.set_ylabel('Normalized Y')
    ax.set_title('Eye Gaze Scatter Plot')

    # Display the scatter plot using Streamlit
    st.pyplot(fig)

    # Display heatmap for top_right.csv
    display_heatmap('top_right.csv', 'Top Right Heatmap')

    # Display heatmap for bottom_left.csv
    display_heatmap('bottom_left.csv', 'Bottom Left Heatmap')

    display_heatmap('eye_gaze_coordinates.csv', 'Eye Gaze Heatmap')

    #display scatter plot of timestamp and score in eye_gaze_coordinates.csv
    df = pd.read_csv('eye_gaze_coordinates.csv')
    scatter_data = df[['Timestamp', 'Score']]
    fig, ax = plt.subplots()
    ax.scatter(scatter_data['Timestamp'], scatter_data['Score'])
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Score')
    ax.set_title('Score Scatter Plot')
    st.pyplot(fig)

    #display blinking scatter plot
    df = pd.read_csv('eye_gaze_coordinates.csv')
    scatter_data = df[['Timestamp', 'Blinks']]
    fig, ax = plt.subplots()
    ax.scatter(scatter_data['Timestamp'], scatter_data['Blinks'])
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Blinking Rate')
    ax.set_title('Blinking Rate Scatter Plot')
    st.pyplot(fig)

    #get average score from csv
    df = pd.read_csv('eye_gaze_coordinates.csv')
    score = df['Score'].mean()
    st.title("Your Study Session Productivity Score")
    st.markdown(score)
    if(score < 0):
        image = Image.open('dead_pokemon.webp')
        st.image(image, caption='Your pokemon died!')
    else:
        image = Image.open('alive_pokemon.png')
        st.image(image, caption='Your pokemon is alive!')

    client = fitbit.Fitbit(
        '23RFVG', 
        'ed6085c8a0e2a7cb173e95e1f97ab6c2',
        access_token='eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1JGVkciLCJzdWIiOiJCUURZOFoiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJ3aHIgd251dCB3cHJvIHdzbGUgd3dlaSB3c29jIHdzZXQgd2FjdCB3bG9jIiwiZXhwIjoxNzAwNDM0MTUzLCJpYXQiOjE3MDA0MDUzNTN9.8JhymbXAm6Yb0sM3ufyNCB5-9Go9cv-PRgYB9fAEUoc', 
        refresh_token='81d7e2d5822b4cdd585f086a1580d79c031674ea8e228bdb83c65bd26c1ed1b3'
    )

    start_date = arrow.get("2023-08-01")
    end_date = arrow.get("2023-12-31")

    # Create a series of 100-day date-range tuples between start_date and end_date
    date_ranges = []
    start_range = start_date
    while start_range < end_date:
        if start_range.shift(days=100) < end_date:
            date_ranges.append((start_range, start_range.shift(days=100)))
            start_range = start_range.shift(days=101)
        else:
            date_ranges.append((start_range, end_date))
            start_range = end_date

    # Print the result to the console
    all_data = []
    heart_data = []
    for date_range in date_ranges:
        print(f"Requesting data for {date_range[0]} to {date_range[1]}.")
        url = f"{client.API_ENDPOINT}/1.2/user/-/sleep/date/{date_range[0].year}-{date_range[0].month:02}-{date_range[0].day:02}/{date_range[1].year}-{date_range[1].month:02}-{date_range[1].day:02}.json"
        heartrateUrl = f"{client.API_ENDPOINT}/1/user/-/activities/heart/date/{date_range[0].year}-{date_range[0].month:02}-{date_range[0].day:02}/{date_range[1].year}-{date_range[1].month:02}-{date_range[1].day:02}/15min.json"
        range_data = client.make_request(url)
        heartData = client.make_request(heartrateUrl)
        all_data.append(range_data)
        heart_data.append(heartData)
        print(f"Success!")
    sleep_summaries = []
    print(heart_data)

    # Iterate through all data and create a list of dictionaries of results:
    for data in all_data:
        for sleep in data["sleep"]:
            # For simplicity, ignoring "naps" and going for only "stage" data
            if sleep["isMainSleep"] and sleep["type"] == "stages":
                sleep_summaries.append(dict(
                date=pd.to_datetime(sleep["dateOfSleep"]).date(),
                duration_hours=sleep["duration"]/1000/60/60,
                total_sleep_minutes=sleep["minutesAsleep"],
                total_time_in_bed=sleep["timeInBed"],
                start_time=sleep["startTime"],
                deep_minutes=sleep["levels"]["summary"].get("deep").get("minutes"),
                light_minutes=sleep["levels"]["summary"].get("light").get("minutes"),
                rem_minutes=sleep["levels"]["summary"].get("rem").get("minutes"),
                wake_minutes=sleep["levels"]["summary"].get("wake").get("minutes"),            
            ))
    # Convert new dictionary format to DataFrame
    sleep_data = pd.DataFrame(sleep_summaries)
    # Sort by date and view first rows
    sleep_data.sort_values("date", inplace=True)
    sleep_data.reset_index(drop=True, inplace=True)
    print(sleep_data.head())
    # It's useful for grouping to get the "date" from every timestamp
    sleep_data["date"] = pd.to_datetime(sleep_data["date"])
    # Also add a boolean column for weekend detection
    sleep_data["is_weekend"] = sleep_data["date"].dt.weekday > 4
    # Sleep distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    (sleep_data["total_sleep_minutes"]/60).plot(
        kind="hist", 
        bins=50, 
        alpha=0.8,
        ax=ax
    )
    (sleep_data["total_time_in_bed"]/60).plot(
        kind="hist", 
        bins=50, 
        alpha=0.8
    )
    plt.legend()
    # add some nice axis labels:
    ax = plt.gca()
    ax.set_xticks(range(2,12))
    plt.grid("minor", linestyle=":")
    plt.xlabel("Hours")
    plt.ylabel("Frequency")
    plt.title("Sleeping Hours")
    st.pyplot(fig)
    #Plot a scatter plot directly from Pandas
    fig, ax = plt.subplots(figsize=(10, 10))
    sleep_data.plot(
        x="total_time_in_bed", 
        y="total_sleep_minutes", 
        kind="scatter", 
        ax=ax
    )
    # Add a perfect 1:1 line for comparison
    ax = plt.gca()
    ax.set_aspect("equal")
    x = np.linspace(*ax.get_xlim())
    ax.plot(x,x, linestyle="--")
    plt.grid(linestyle=":")
    st.pyplot(fig)
    # Sleep makeup - calculate data to plot
    plot_data = sleep_data.\
        sort_values("date").\
        set_index("date")\
        [["deep_minutes", "light_minutes", "rem_minutes", "wake_minutes"]]
    # Matplotlib doesn't natively support stacked bars, so some messing here:
    df = plot_data
    fig, ax = plt.subplots(figsize=(30,7), constrained_layout=True)
    bottom = 0
    for c in df.columns:
        ax.bar(df.index, df[c], bottom=bottom, width=1, label=c)
        bottom+=df[c]
    # Set a date axis for the x-axis allows nicer tickmarks.
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.legend()
    plt.xlabel("Date")
    plt.ylabel("Minutes")
    # Show a subset of data for clarity on the website:
    plt.xlim(pd.to_datetime("2023-08-01"), pd.to_datetime("2023-12-31"))
    st.pyplot(fig)

    # Heart Rate
    heart_summaries = []
    for data in heart_data:
        for heart in data["activities-heart"]:
            print(heart)
            # For simplicity, ignoring "naps" and going for only "stage" data
            if "restingHeartRate" in heart["value"] and heart["value"]["restingHeartRate"]:
                heart_summaries.append(dict(
                    date=pd.to_datetime(heart["dateTime"]).date(),
                    resting_heart_rate=heart["value"]["restingHeartRate"]
            ))
    # Convert new dictionary format to DataFrame
    heart_data = pd.DataFrame(heart_summaries)
    # Sort by date and view first rows
    heart_data.sort_values("date", inplace=True)
    heart_data.reset_index(drop=True, inplace=True)
    print(heart_data.head())
    # It's useful for grouping to get the "date" from every timestamp
    heart_data["date"] = pd.to_datetime(heart_data["date"])
    # Also add a boolean column for weekend detection
    heart_data["is_weekend"] = heart_data["date"].dt.weekday > 4
    plot_data = heart_data.\
        sort_values("date").\
        set_index("date")\
        [["resting_heart_rate"]]
    # Matplotlib doesn't natively support stacked bars, so some messing here:
    df = plot_data
    fig, ax = plt.subplots(figsize=(30,7), constrained_layout=True)
    bottom = 0
    for c in df.columns:
        ax.bar(df.index, df[c], bottom=bottom, width=1, label=c)
        bottom+=df[c]
    # Set a date axis for the x-axis allows nicer tickmarks.
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.legend()
    plt.xlabel("Date")
    plt.ylabel("Resting Heart Rate")
    # Show a subset of data for clarity on the website:
    plt.xlim(pd.to_datetime("2023-08-01"), pd.to_datetime("2023-12-31"))
    st.pyplot(fig)


    # sameDay = f"{client.API_ENDPOINT}/1/user/-/activities/heart/date/today/today/1min.json"
    # sameDayData = client.make_request(sameDay)
    # heart_summaries = []
    # for heart in sameDayData["activities-heart"]:
    #   # print(heart)
    #   # For simplicity, ignoring "naps" and going for only "stage" data
    #   if "restingHeartRate" in heart["value"] and heart["value"]["restingHeartRate"]:
    #     heart_summaries.append(dict(
    #         date=pd.to_datetime(heart["dateTime"]).date(),
    #         resting_heart_rate=heart["value"]["restingHeartRate"]
    #     ))
    # Convert new dictionary format to DataFrame
    # heart_data = pd.DataFrame(heart_summaries)
    # # Sort by date and view first rows
    # heart_data.sort_values("date", inplace=True)
    # heart_data.reset_index(drop=True, inplace=True)
    # print(heart_data.head())
    # # It's useful for grouping to get the "date" from every timestamp
    # heart_data["date"] = pd.to_datetime(heart_data["date"])
    # # Also add a boolean column for weekend detection
    # heart_data["is_weekend"] = heart_data["date"].dt.weekday > 4
    # plot_data = heart_data.\
    #   sort_values("date").\
    #   set_index("date")\
    #   [["resting_heart_rate"]]
    # # Matplotlib doesn't natively support stacked bars, so some messing here:
    # df = plot_data
    # fig, ax = plt.subplots(figsize=(30,7), constrained_layout=True)
    # bottom = 0
    # for c in df.columns:
    #   ax.bar(df.index, df[c], bottom=bottom, width=1, label=c)
    #   bottom+=df[c]
    # # Set a date axis for the x-axis allows nicer tickmarks.
    # ax.xaxis.set_major_locator(mdates.MonthLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # ax.legend()
    # plt.xlabel("Date")
    # plt.ylabel("Resting Heart Rate")
    # # Show a subset of data for clarity on the website:
    # plt.xlim(pd.to_datetime("2023-08-01"), pd.to_datetime("2023-12-31"))
    # plt.show()

    sameDay = f"{client.API_ENDPOINT}/1/user/-/activities/heart/date/today/1d/1min.json?timezone=EST"
    sameDayData = client.make_request(sameDay)
    # print(sameDayData["activities-heart-intraday"]["dataset"])
    #graph it
    heart_summaries = []
    for heart in sameDayData["activities-heart-intraday"]["dataset"]:
    # print(heart)
    # For simplicity, ignoring "naps" and going for only "stage" data
        if "value" in heart:
            heart_summaries.append(dict(
                time=heart["time"],
                heart_rate=heart["value"]
        ))
    # Convert new dictionary format to DataFrame
    heart_data = pd.DataFrame(heart_summaries)
    # Sort by date and view first rows
    heart_data.sort_values("time", inplace=True)
    heart_data.reset_index(drop=True, inplace=True)
    print(heart_data.head())
    #scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    heart_data.plot(
        x="time", 
        y="heart_rate", 
        kind="scatter", 
        ax=ax
    )
    # make x-axis time values readable
    ax = plt.gca()
    ax.set_xticks(range(0,1440,60))
    ax.set_xticklabels(range(0,24))
    plt.xlabel("Time")
    plt.grid(linestyle=":")
    # plt.show()
    st.pyplot(fig)


    # devices = f"{client.API_ENDPOINT}/1/user/-/devices.json"
    # devicesData = client.make_request(devices)
    # print(devicesData)
    # deviceID = devicesData[0].get("id")

    #create alarm
    # alarm = f"{client.API_ENDPOINT}/1/user/-/devices/tracker/{deviceID}/alarms.json"
    # time = "01:00"
    # enabled = True
    # weekDays = "SUNDAY"
    # recurring = False
    # alarm += f"?time={time}&enabled={enabled}&weekDays={weekDays}&recurring={recurring}"
    # alarmData = client.make_request(alarm)
    # print(alarmData)
    load_dotenv()
    # client = OpenAI()

    # completion = client.chat.completions.create(
    #   model="gpt-4-1106-preview",
    #   messages=[
    #     {"role": "system", "content": "You are a software helping people with productivty. Give tips for how to help people be more productive."},
    #   ]
    # )

    # st.markdown(completion.choices[0].message.content)
    # print("DONE")
    client = AzureOpenAI(
        azure_endpoint = 'https://api.umgpt.umich.edu/azure-openai-api/ptu',
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2023-03-15-preview"
    )
    response = client.chat.completions.create(
        model="gpt-4", # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a software helping people with productivty. Give tips for how to help people be more productive."},
        ]
    )
    st.title("Azure OpenAI Chatbot Feedback on Your Study Session Productivity")
    st.markdown(response.choices[0].message.content)

    cred_obj = firebase_admin.credentials.Certificate('alert-563fb-firebase-adminsdk-bssfd-87718064f0.json')
    default_app = firebase_admin.initialize_app(cred_obj, 
        {
        'databaseURL':'https://alert-563fb-default-rtdb.firebaseio.com/',
        'storageBucket':'alert-563fb.appspot.com'
        })
    fileName = "eye_gaze_coordinates.csv"
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)


def main():
    # Set page configuration to avoid script rerun on every interaction
    st.set_page_config(page_title="Multi-Page App", page_icon="🅰️")
    
    

    if "page" not in st.session_state:
        st.session_state.page = "analyze"
    if st.session_state.page == "track":
        page_track()
    elif st.session_state.page == "analyze":
        page_analyze()
    

if __name__ == "__main__":
    main()