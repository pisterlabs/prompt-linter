import time
import flet as ft
from tts import tts
from plutox import *

class Message():
    def __init__(self, user_name: str, text: str, message_type: str):
        self.user_name = user_name
        self.text = text
        self.message_type = message_type

class ChatMessage(ft.Row):
    def __init__(self, message: Message):
        super().__init__()
        self.vertical_alignment="start"
        self.controls=[
                ft.CircleAvatar(
                    content=ft.Text(self.get_initials(message.user_name)),
                    color=ft.colors.WHITE,
                    bgcolor=self.get_avatar_color(message.user_name),
                ),
                ft.Column(
                    [
                        ft.Text(message.user_name, weight="bold"),
                        ft.Text(message.text, selectable=True),
                    ],
                    tight=True,
                    spacing=5,
                ),
            ]

    def get_initials(self, user_name: str):
        return user_name[:1].capitalize()

    def get_avatar_color(self, user_name: str):
        colors_lookup = [
            ft.colors.AMBER,
            ft.colors.BLUE,
            ft.colors.BROWN,
            ft.colors.CYAN,
            ft.colors.GREEN,
            ft.colors.INDIGO,
            ft.colors.LIME,
            ft.colors.ORANGE,
            ft.colors.PINK,
            ft.colors.PURPLE,
            ft.colors.RED,
            ft.colors.TEAL,
            ft.colors.YELLOW,
        ]
        return colors_lookup[hash(user_name) % len(colors_lookup)]

def main(page: ft.Page):
    page.horizontal_alignment = "stretch"
    page.title = "STAMP Support"

    def join_chat_click(e):
        if not join_user_name.value:
            join_user_name.error_text = "Name cannot be blank!"
            join_user_name.update()
        else:
            if join_user_name.value=="3511plutox":
                page.session.set("user_name", "Admin")
                page.dialog.open = False
                new_message.prefix = ft.Text(f"{join_user_name.value}: ")
                page.pubsub.send_all(Message(user_name=join_user_name.value, text=f"Admin has joined the chat.", message_type="login_message"))
                page.update()
            elif join_user_name.value=="Admin":
                join_user_name.error_text = "Name cannot be Admin"
                join_user_name.update()
            else:
                page.session.set("user_name", join_user_name.value)
                page.dialog.open = False
                new_message.prefix = ft.Text(f"{join_user_name.value}: ")
                page.pubsub.send_all(Message(user_name=join_user_name.value, text=f"{join_user_name.value} has joined the chat.", message_type="login_message"))
                page.update()

    def send_message_click(e):
        if new_message.value != "":
            page.pubsub.send_all(Message(page.session.get("user_name"), new_message.value, message_type="chat_message"))
            temp=new_message.value
            new_message.value = ""
            new_message.focus()    
            if temp.startswith("/?"):
                res=chatgpt(temp)
                if len(res) > 220: # adjust the maximum length as needed
                    res = '\n'.join([res[i:i+220] for i in range(0, len(res), 220)])
                page.pubsub.send_all(Message("STAMP Support", res, message_type="chat_message"))
                tts(res)
            elif page.session.get("user_name")=="Admin":
                if temp=="?spinall" or temp=="take off":
                    res="PlutoX Takeoff Instantiated"
                    page.pubsub.send_all(Message("Pluto X", res, message_type="chat_message"))
                    tts(res)
                    spinall()
                elif temp=="?backward" or temp=="backward":
                    res="PlutoX Backward Motion Instantiated"
                    page.pubsub.send_all(Message("Pluto X", res, message_type="chat_message"))
                    tts(res)
                    backward()
                elif temp=="?forward" or temp=="forward":
                    res="PlutoX Forward Motion Instantiated"
                    page.pubsub.send_all(Message("Pluto X", res, message_type="chat_message"))
                    tts(res)
                    forward()
                elif temp=="?left" or temp=="left":
                    res="PlutoX Left Motion Instantiated"
                    page.pubsub.send_all(Message("Pluto X", res, message_type="chat_message"))
                    tts(res)
                    left()
                elif temp=="?right" or temp=="right":
                    res="PlutoX Right Motion Instantiated"
                    page.pubsub.send_all(Message("Pluto X", res, message_type="chat_message"))
                    tts(res)
                    right()
                elif temp=="?m1" or temp=="M1":
                    res="PlutoX M1 Propeller Instantiated"
                    page.pubsub.send_all(Message("Pluto X", res, message_type="chat_message"))
                    tts(res)
                    m1()
                elif temp=="?m2" or temp=="M2":
                    res="PlutoX M2 Propeller Instantiated"
                    page.pubsub.send_all(Message("Pluto X", res, message_type="chat_message"))
                    tts(res)
                    m2()
                elif temp=="?m3" or temp=="M3":
                    res="PlutoX M3 Propeller Instantiated"
                    page.pubsub.send_all(Message("Pluto X", res, message_type="chat_message"))
                    tts(res)
                    m3()
                elif temp=="?m4" or temp=="M4":
                    res="PlutoX M4 Propeller Instantiated"
                    page.pubsub.send_all(Message("Pluto X", res, message_type="chat_message"))
                    tts(res)
                    m4()
                temp=""
            else:
                pass
            page.update()
           
    def spinall():
        client = Drone()
        client.arm()
        time.sleep(5)
        client.disArm()

    def backward():
        client = Drone()
        client.backward()
        time.sleep(5)
        client.backwardstop()

    def forward():
        client = Drone()
        client.forward()
        time.sleep(5)
        client.forwardstop()
        
    def right():
        client = Drone()
        client.right()
        time.sleep(5)
        client.rightstop()
        
    def left():
        client = Drone()
        client.left()
        time.sleep(5)
        client.leftstop()

    def m1():
        client = Drone()
        client.m1()
        time.sleep(5)
        client.m1stop()  
             
    def m2():
        client = Drone()
        client.m2()
        time.sleep(5)
        client.m2stop() 
              
    def m3():
        client = Drone()
        client.m3()
        time.sleep(5)
        client.m3stop() 
              
    def m4():
        client = Drone()
        client.m4()
        time.sleep(5)
        client.m4stop()       

        
    def chatgpt(message):
        import openai

        # Set up the OpenAI API client
        openai.api_key = ""

        # Set up the model and prompt
        model_engine = "text-davinci-003"
        # prompt = "Can you provide information about drones and their capabilities? : " + message
        prompt=message
        print(prompt)
        # Generate a response
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        response = completion.choices[0].text.strip()
        if response.startswith('\n'):
            response = response[1:]
        return response

        
    def on_message(message: Message):
        if message.message_type == "chat_message":
            m = ChatMessage(message)
        elif message.message_type == "login_message":
            m = ft.Text(message.text, italic=True, color=ft.colors.BLACK45, size=12)
        chat.controls.append(m)
        page.update()

    page.pubsub.subscribe(on_message)

    # A dialog asking for a user display name
    join_user_name = ft.TextField(
        label="Enter your name to join the chat",
        autofocus=True,
        on_submit=join_chat_click,
    )
    page.dialog = ft.AlertDialog(
        open=True,
        modal=True,
        title=ft.Text("Welcome!"),
        content=ft.Column([join_user_name], width=300, height=70, tight=True),
        actions=[ft.ElevatedButton(text="Join chat", on_click=join_chat_click)],
        actions_alignment="end",
    )

    # Chat messages
    chat = ft.ListView(
        expand=True,
        spacing=10,
        auto_scroll=True,
    )

    # A new message entry form
    new_message = ft.TextField(
        hint_text="Write a message...",
        autofocus=True,
        shift_enter=True,
        min_lines=1,
        max_lines=5,
        filled=True,
        expand=True,
        on_submit=send_message_click,
    )

    # Add everything to the page
    page.add(
        ft.Container(
            content=chat,
            border=ft.border.all(1, ft.colors.OUTLINE),
            border_radius=5,
            padding=10,
            expand=True,
        ),
        ft.Row(
            [
                new_message,
                ft.IconButton(
                    icon=ft.icons.SEND_ROUNDED,
                    tooltip="Send message",
                    on_click=send_message_click,
                ),
            ]
        ),
    )

ft.app(port=8550, target=main, view=ft.WEB_BROWSER)
# ft.app(target=main)