import streamlit as st
import pywhatkit
import pyttsx3
import webbrowser
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import subprocess
import twilio
import smtplib
import os
import time
import plyer
import cv2
import cvzone
from contextlib import contextmanager
import streamlit as st
from transformers import pipeline
import numpy as np
import openai
import io
import traceback
from PIL import Image


def main():
        

    menu_options = [
        "Docker Terminal",
        "Linux Terminal",
        "CHIPITI",
        "Computer Vision",
        "Automated ML","Python Interpreter","Send Whatsapp Message","Text to Speech","Open any website",
        "SMS Sender","Email Sender","Create a NN with Gesture","Sentiment Analysis"
        
       
    ]
    st.sidebar.title("TEAM 9")
    choice = st.sidebar.selectbox("Select an option", menu_options)
    
    if choice == menu_options[0]:
        option1()
    elif choice == menu_options[1]:
        option2()
    elif choice == menu_options[2]:
        option3()
    elif choice == menu_options[3]:
        option4()
    elif choice == menu_options[4]:
        automl()
    elif choice == menu_options[5]:
        option5()
    elif choice == menu_options[6]:
        option6()
    elif choice == menu_options[7]:
        option7()
    elif choice == menu_options[8]:
        option8()
    elif choice == menu_options[9]:
        option9()
    elif choice == menu_options[10]:
        option10()
    elif choice == menu_options[11]:
        option11()
    elif choice == menu_options[12]:
        option12()    
    


def option1():



    # Function to execute Docker commands using the CGI script
    def execute_docker_command(command):
        cgi_url = "http://65.0.76.20/cgi-bin/docker_cg1.py"
        response = requests.post(cgi_url, data={"command": command})
        return response.text

    # Streamlit app
    def main():
        
            

        st.title("Docker Admin")

        # Docker commands as buttons
        if st.button("List Docker Images"):
            result = execute_docker_command("docker images")
            st.subheader("Docker Images:")
            st.text(result)

        if st.button("List Docker Containers"):
            result = execute_docker_command("docker ps")
            st.subheader("Docker Containers:")
            st.text(result)

        st.subheader("Pull Docker Image")
        image_name = st.text_input("Enter the Docker image name:")
        if st.button("Pull"):
            command = f"docker pull {image_name}"
            result = execute_docker_command(command)
            st.subheader("Command Output:")
            st.text(result)
        
        st.subheader("Custom Command")
        custom_command = st.text_input("Enter Custom Docker Command:")
        if st.button("Execute Custom Command"):
            result = execute_docker_command(custom_command)
            st.subheader("Command Output:")
            st.text(result)

    if __name__ == "__main__":
        main()


def option2():
    # Function to execute Linux commands using the CGI script
    def execute_linux_command(command):
        cgi_url = "http://65.0.76.20/cgi-bin/linux.py"
        response = requests.post(cgi_url, data={"command": command})
        return response.text

    # Streamlit app
    def main():
        st.title("Linux Terminal")

        # Sidebar with options
        st.sidebar.subheader("Options")
        command_options = ["ls", "pwd", "date", "ifconfig", "Custom Command"]
        selected_option = st.sidebar.selectbox("Select a command:", command_options)

        if selected_option == "Custom Command":
            command = st.text_input("Enter a Linux command:")
        else:
            command = selected_option

        if st.sidebar.button("Execute"):
            result = execute_linux_command(command)
            st.subheader("Command Output:")
            st.code(result, language="shell")

    if __name__ == "__main__":
        main()

def option3():


    openai.api_key = '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'

    # Function to generate response using the GPT model
    def generate_response(prompt):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=50, 
        )
        return response.choices[0].text.strip()

    def main():
        st.title("Technology Learning Portal")

        st.write("Enter your question or topic below to learn about technologies:")

        user_input = st.text_area("Your Question:")

        if st.button("Get Answer"):
            if user_input:
                response = generate_response(user_input)
                st.subheader("Answer:")
                engine = pyttsx3.init()

                st.write(response)
                engine.say(response)

                engine.runAndWait()
                
            else:
                st.warning("Please enter a question or topic.")

        

    if __name__ == "__main__":
        main()

def option4():
    if st.button("Use Goggles"):
        def main():
            run = st.checkbox('Run')
            FRAME_WINDOW = st.image([])
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            accessories_image = cv2.imread("pngwing.com.png", cv2.IMREAD_UNCHANGED)
            cap = cv2.VideoCapture(0)
            
            while run:
                _, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    resized_sunglasses = cv2.resize(accessories_image, (w, h))
                    for i in range(resized_sunglasses.shape[0]):
                        for j in range(resized_sunglasses.shape[1]):
                            if resized_sunglasses[i, j, 3] != 0:
                                frame[y + i, x + j] = resized_sunglasses[i, j, :3]
                FRAME_WINDOW.image(frame)
            else:
                st.write('Stopped')
        if __name__ == "__main__":
            main()


    if st.button("AWS Services"):
        def main():
            run = st.checkbox('Run')
            import cv2
            from cvzone.HandTrackingModule import HandDetector
            import boto3
            FRAME_WINDOW = st.image([])
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)
            ec2 = boto3.resource("ec2")

            allOs = []

            def myOSLaunch():
                instances = ec2.create_instances(
                        ImageId="ami-0a2acf24c0d86e927",
                        MinCount=1,
                        MaxCount=1,
                        InstanceType="t2.micro",
                        SecurityGroupIds=["sg-02a2f3ad81863794f"]
                    )
                myId = instances[0].id
                allOs.append(myId)
                print("Total Number of OS:", len(allOs))
                print(myId)

            def osTerminate():
                osDelete = allOs.pop()
                ec2.instances.filter(InstanceIds=[osDelete]).terminate()
                print(osDelete)
                print("Total number of instances:", len(allOs))

            def endAllInstances():
                while True:
                    if allOs == []:
                        break
                    osDelete = allOs.pop()
                    print(osDelete)
                    ec2.instances.filter(InstanceIds=[osDelete]).terminate()

            detector = HandDetector(maxHands=1)
            while run:
                _, photo = cap.read()
                hand = detector.findHands(photo, draw=False)
                cv2.imshow("photo", photo)
                #   print(cv2.waitKey(100))
                if cv2.waitKey(1000) == 13:
                    break
                if hand:
                    lmlist = hand[0]
                    noOfFingers = detector.fingersUp(lmlist)
                    if noOfFingers == [0,1,0,0,0]:
                        print("Index Finger")
                        myOSLaunch()
                    if noOfFingers == [1,0,0,0,0]:
                        print("Thumb")
                        osTerminate()
            FRAME_WINDOW.image(photo)            
            endAllInstances()
            cv2.destroyAllWindows()
            cap.release()
        if __name__ == "__main__":
            main()


    if st.button("Face Blur"):
        def main():
            run = st.checkbox('Run')
            FRAME_WINDOW = st.image([])
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            video_capture = cv2.VideoCapture(0)  

            while run:
                _, frame = video_capture.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                blur = cv2.GaussianBlur(frame, (99, 99), 0)

                for (x, y, w, h) in faces:

                    left = frame[:, 0:x]  
                    blur_roi = blur[:, 0:x] 
                    frame[:, 0:x] = blur_roi


                    right = frame[:, x+w:]  
                    blur_roi = blur[:, x+w:] 
                    frame[:, x+w:] = blur_roi

                    upper = frame[0:y, :]  
                    blur_roi = blur[0:y, :] 
                    frame[0:y, :] = blur_roi

                    lower = frame[y+h:, :]  
                    blur_roi = blur[y+h:, :] 
                    frame[y+h:, :] = blur_roi
                FRAME_WINDOW.image(frame)
            else:
                st.write('Stopped')
        if __name__ == "__main__":
            main()

    if st.button("FaceDistance"):
        def main():
            run = st.checkbox('Run')
            FRAME_WINDOW = st.image([])        
            KNOWN_FACE_WIDTH = 15.0  
            FOCAL_LENGTH = 700.0     

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            video_capture = cv2.VideoCapture(0)  

            while run:
                _, frame = video_capture.read()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:


                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    face_width_pixels = w
                    distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / face_width_pixels

                    distance_text = f"Distance: {distance:.2f} cm"
                    cv2.putText(frame, distance_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                FRAME_WINDOW.image(frame)
        
        if __name__ == "__main__":
            main()

def option5():
    

    def execute_code(code):
        # Capture standard output
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        try:
            exec(code)
        except Exception:
            st.error(traceback.format_exc())

        # Reset standard output
        sys.stdout = old_stdout

        return redirected_output.getvalue()

    def main():
        st.title("Python Interpreter")

        # Text area for user input
        code = st.text_area("Enter your Python code below:", height=300)

        # Execute button
        if st.button("Execute"):
            if code.strip():
                output = execute_code(code)
                st.code(output, language="python")
            else:
                st.warning("Please enter some Python code.")

    if __name__ == "__main__":
        main()


def option6():
    st.header("Send WhatsApp message")
    # Add your code for Option 2 here
    no = st.text_input("Enter 10 Digit number")
    message = st.text_input("Enter message:")
    if st.button("Send Message"):
        if no and message:
            no = "+91" + no
            pywhatkit.sendwhatmsg_instantly(no, message)
            st.success("Message sent successfully!")

def option7():
    st.header("Convert Text into Speech")
    # Add your code for Option 3 here
    message = st.text_input("Enter message to speak:")
    if st.button("Speak"):
        if message:
            myspeaker = pyttsx3.init()
            myspeaker.say(message)
            myspeaker.runAndWait()
            st.success("Text converted to speech!")

def option8():
    st.header("Open any website URL")
    # Add your code for Option 4 here
    url = st.text_input("Enter the URL:")
    if st.button("Open Website"):
        if url:
            url = "https://" + url
            webbrowser.open(url)
            st.success("Website opened successfully!")

def option9():
    from twilio.rest import Client

    # Replace these with your Twilio credentials
    TWILIO_ACCOUNT_SID = 'ACcdd047fc0ae4aaeb4bca247ad246d010'
    TWILIO_AUTH_TOKEN = 'da445f4cc08b4552d0a508d92261b060'
    TWILIO_PHONE_NUMBER = '+12348039770'

    def send_sms(phone_number, message):
        try:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            client.messages.create(
                to=phone_number,
                from_=TWILIO_PHONE_NUMBER,
                body=message
            )
            return True
        except Exception as e:
            st.error(f"Failed to send SMS. Error: {str(e)}")
            return False

    def main():
        st.title("Automatic Text SMS Sender")

        phone_number = st.text_input("Enter recipient's phone number:")
        message = st.text_area("Enter your message:")

        if st.button("Send SMS"):
            if phone_number and message:
                st.info("Sending SMS...")
                if send_sms(phone_number, message):
                    st.success("SMS sent successfully!")
                else:
                    st.error("Failed to send SMS. Please check the phone number and Twilio credentials.")
            else:
                st.warning("Please enter a valid phone number and message.")

    if __name__ == "__main__":
        main()

def option10():
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    # Gmail credentials
    GMAIL_EMAIL = 'jainabhi7374@gmail.com'
    GMAIL_PASSWORD = 'abhi@2411'

    def send_email(subject, recipients, message):
        try:
            msg = MIMEMultipart()
            msg['From'] = GMAIL_EMAIL
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject

            body = message
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(GMAIL_EMAIL, GMAIL_PASSWORD)
            server.sendmail(GMAIL_EMAIL, recipients, msg.as_string())
            server.quit()

            return True
        except Exception as e:
            st.error(f"Failed to send email. Error: {str(e)}")
            return False

    def main():
        st.title("Automatic Email Sender")

        recipients = st.text_input("Enter recipient(s) email address (comma-separated):")
        subject = st.text_input("Enter email subject:")
        message = st.text_area("Enter your message:")

        if st.button("Send Email"):
            if recipients and subject and message:
                recipient_list = [email.strip() for email in recipients.split(',')]
                st.info("Sending Email...")
                if send_email(subject, recipient_list, message):
                    st.success("Email sent successfully!")
                else:
                    st.error("Failed to send email. Please check your Gmail credentials.")
            else:
                st.warning("Please enter valid recipient email address(es), subject, and message.")

    if __name__ == "__main__":
        main()

    
def automl():
    st.title("Automated MachineLearning Model")
    
    # File path input
    file = st.text_input("Enter the file path:")
    
    if not file:
        st.warning("Please enter a file path.")
        return
    
    try:
        data = pd.read_csv(file)
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
        return
    
    a = st.text_input("Enter the X labels (space-separated):")
    a = a.split()
    x = data[a]
    
    b = st.text_input("Enter the Y label:")
    y = data[b]
    
    if len(a) == 1:
        x = x.values.reshape(len(x), 1)
        model = LinearRegression()
        model.fit(x, y)
        st.write("Model Coefficients:", model.coef_)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
        model = LinearRegression()
        model.fit(x_train, y_train)
        st.write("Model Coefficients:", model.coef_)

    automl()

def option11():
    
        
    if st.button("Run"):

        def main():
            run=st.checkbox("Run")
            FRAME_WINDOW = st.image([])        
            import cv2
            from cvzone.HandTrackingModule import HandDetector
            from tensorflow import keras
            from tensorflow.keras import layers
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            num_conv_layers = 0
            kernel_size = (1, 1)
            pool_size = (0, 0)
            num_connected_layers = 0
            num_classes = 0
            def build_cnn(num_conv_layers, kernel_size, pool_size, num_connected_layers, num_classes, num_filters=32):
                # Function body
                model = keras.Sequential()
                
                # Add input convolutional layer
                model.add(layers.Conv2D(num_filters, kernel_size, activation='relu', input_shape=(64, 64, 3)))
                
                # Add additional convolutional layers
                for _ in range(num_conv_layers - 1):
                    model.add(layers.Conv2D(num_filters, kernel_size, activation='relu'))
                
                # Add pooling layer
                model.add(layers.MaxPooling2D(pool_size=(1,1)))
                
                # Flatten the feature maps
                model.add(layers.Flatten())
                
                # Add fully connected layers
                for _ in range(num_connected_layers):
                    model.add(layers.Dense(20, activation='relu'))
                
                # Add output layer
                model.add(layers.Dense(num_classes, activation='softmax'))
                
                return model



            while run:
                _, photo = cap.read()
                cv2.imshow("photo", photo)
                if cv2.waitKey(1) == 13:
                    break
                hand = detector.findHands(photo, draw=False)
                if hand:
                    lmlist = hand[0]
                    noOfFingers = detector.fingersUp(lmlist)
                    if noOfFingers == [0,1,0,0,0]:
                        cv2.putText(photo, "Num Conv Layers: ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("photo", photo)
                        if cv2.waitKey(1) == 13:
                            break
                        if hand:
                            
                            lmlist = hand[0]
                            noOfFingers = detector.fingersUp(lmlist)
                            if noOfFingers == [0,1,0,0,0]:
                                num_conv_layers=1
                                
                            if noOfFingers == [0,1,1,0,0]:
                                num_conv_layers=2
                                
                            if noOfFingers == [0,1,1,1,0]:
                                num_conv_layers=3
                                
                            if noOfFingers == [0,1,1,1,1]:
                                num_conv_layers=4
                                
                            if noOfFingers == [1,1,1,1,1]:
                                num_conv_layers=5
                                
                            
                    elif noOfFingers == [0,1,1,0,0]:
                        cv2.putText(photo, "Kernel Size: ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("photo", photo)
                        if cv2.waitKey(1) == 13:
                            break
                        if hand:
                            lmlist = hand[0]
                            noOfFingers = detector.fingersUp(lmlist)
                            if noOfFingers == [0,1,1,0,0]:
                                kernel_size=(2,2)
                                
                            if noOfFingers == [0,1,1,1,0]:
                                kernel_size=(3,3)
                                
                            if noOfFingers == [0,1,1,1,1]:
                                kernel_size=(4,4)
                                
                            if noOfFingers == [1,1,1,1,1]:
                                kernel_size=(5,5)
                                
                        

                    elif noOfFingers == [0,1,1,1,0]:
                        cv2.putText(photo, "Pool Size: ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("photo", photo)
                        if cv2.waitKey(1) == 13:
                            break
                        if hand:
                            lmlist = hand[0]
                            noOfFingers = detector.fingersUp(lmlist)
                            if noOfFingers == [0,1,0,0,0]:
                                pool_size=(1,1)
                                
                            if noOfFingers == [0,1,1,0,0]:
                                pool_size=(2,2)
                                
                            if noOfFingers == [0,1,1,1,0]:
                                pool_size=(3,3)
                                
                            if noOfFingers == [0,1,1,1,1]:
                                pool_size=(4,4)
                                
                            if noOfFingers == [1,1,1,1,1]:
                                pool_size=(5,5)
                                
                                
                    elif noOfFingers == [0,1,1,1,1]:
                        cv2.putText(photo, "Connected Layers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("photo", photo)
                        if cv2.waitKey(1) == 13:
                            break
                        if hand:
                            lmlist = hand[0]
                            noOfFingers = detector.fingersUp(lmlist)
                            if noOfFingers == [0,1,0,0,0]:
                                num_connected_layers=1
                                
                            if noOfFingers == [0,1,1,0,0]:
                                num_connected_layers=2
                                
                            if noOfFingers == [0,1,1,1,0]:
                                num_connected_layers=3
                                
                            if noOfFingers == [0,1,1,1,1]:
                                num_connected_layers=4
                                
                            if noOfFingers == [1,1,1,1,1]:
                                num_connected_layers=5
                                
                        
                    elif noOfFingers == [1,1,1,1,1]:
                        cv2.putText(photo, "Num of classes: ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("photo", photo)
                        if cv2.waitKey(1) == 13:
                            break
                        if hand:
                            lmlist = hand[0]
                            noOfFingers = detector.fingersUp(lmlist)
                            if noOfFingers == [0,1,0,0,0]:
                                num_classes=1

                            if noOfFingers == [0,1,1,0,0]:
                                num_classes=2

                            if noOfFingers == [0,1,1,1,0]:
                                num_classes=3
                                
                    elif noOfFingers == [0,1,0,0,1]:
                        cv2.destroyAllWindows()
                        cap.release()
                        break
                FRAME_WINDOW.image(photo)

            model=build_cnn(num_conv_layers, kernel_size, pool_size, num_connected_layers, num_classes=1, num_filters=32)
            cv2.destroyAllWindows()
            cap.release()
            model.summary(print_fn=st.write)

        if __name__ == "__main__":
            main()

def option12():

    nlp = pipeline("sentiment-analysis")
    def main():
        st.title("Sentiment Analysis App")

        # Get user input for text
        text = st.text_area("Enter some text:")

        if text:
            # Analyze the sentiment of the text
            result = nlp(text)[0]

            # Extract sentiment label and score
            sentiment_label = result["label"]
            sentiment_score = result["score"]

            # Display the result
            st.subheader("Sentiment Analysis Result:")
            st.write(f"Sentiment: {sentiment_label.capitalize()}")
            st.write(f"Confidence Score: {sentiment_score:.2f}")

    if __name__ == "__main__":
        main()























if __name__ == "__main__":
    main()