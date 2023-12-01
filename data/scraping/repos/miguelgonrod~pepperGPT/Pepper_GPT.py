#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import openai
import speech_recognition as sr

openai.api_key="Put here your openAI api key"

class Recognizer:
    messages = [{"role": "system", "content": "You are a kind and helpful assistant"},]
    node = "/speech"
    trainingMessages = ["De ahora en adelante eres pepper, el robot de softbank", "De ahora en adelante si no sabes una respuesta responde \"No se la respuesta, preguntale a mi supervisor\""]
    r = sr.Recognizer()

    def __init__(self):
        pass

    def training(self):
        for message in self.trainingMessages:
            rospy.loginfo(message)
            self.messages.append({"role": "user", "content": message},)
            chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages)
            reply = chat.choices[0].message.content
            rospy.loginfo(reply)
            self.messages.append({"role": "assistant", "content": reply},)

    def voiceRecognition(self):
        with sr.Microphone() as source:
            rospy.loginfo("Di algo...")
            audio = self.r.listen(source)
            try:
                rospy.loginfo("Reconociendo...")
                texto = self.r.recognize_google(audio, language="ES")
                self.messages.append({"role": "user", "content": texto},)
                chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages)
                rospy.loginfo("Texto reconocido: " + texto)
                reply = chat.choices[0].message.content
                rospy.loginfo("ChatGPT: %s", reply)
                self.messages.append({"role": "assistant", "content": reply},)
                return reply

            except sr.UnknownValueError:
                rospy.loginfo("No se pudo reconocer el audio")
                return "Lo siento no te entendí"

            except sr.RequestError as e:
                rospy.loginfo("Error al enviar la solicitud a la API de Google: {0}".format(e))
                return "Lo siento, hubo un error al calcular tu respuesta"

    def pepperPublisher(self):
        pub = rospy.Publisher('/speech', String, queue_size=10)
        rospy.init_node('pepperPub', anonymous=True)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            message = self.voiceRecognition()
            rospy.loginfo(message)
            pub.publish(message)
            rate.sleep()

if __name__ == "__main__":
    recognition = Recognizer()

    recognition.training()
    recognition.pepperPublisher()
