import cv2
import numpy as np
import random
import openai
import pyautogui
import time
import keyboard
import pyperclip

openai.api_key = 'YOUR OPEN AI API KEY'

describe_chatgpt_mission = "Short reply to this Tweet, and refrence that you are searching for python devs. Link in profile: "


def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Choose the appropriate GPT-3 model
        messages=[
            {"role": "user", "content": describe_chatgpt_mission + prompt}
        ],
        max_tokens= 150 # Adjust as needed
    )
    return response.choices[0].message['content'].strip()


def locate_button(template_path, th):

    def locate_image_on_screen(tp, threshold):
        template = cv2.imread(tp, cv2.IMREAD_GRAYSCALE)
        screenshot = pyautogui.screenshot()
        screenshot_gray = cv2.cvtColor(cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            return max_loc
        else:
            return None
    button_loc = locate_image_on_screen(template_path, threshold=th)
    return button_loc

def overview_new_tweets():
    time_active = 6000 #in seconds x10
    iterations = 0
    my_response = []
    while iterations <= time_active:
        iterations += 1
        time.sleep(10)

        button_loc = locate_button('img/new_posts.png', 0.45)
        try:
            if button_loc is not None:
                button_x, button_y = button_loc
                pyautogui.moveTo(button_x + 590, button_y + 10, duration=random.uniform(0.2, 1.5))
                pyautogui.click()
                time.sleep(random.randrange(3, 5))
                pyautogui.click()
                time.sleep(1)
                pyautogui.moveTo(button_x + 590, button_y - 120, duration=random.uniform(0.2, 1.5))
                for click in range(3):
                    pyautogui.click()
                time.sleep(1)
                close = locate_button('img/close_img.png', 0.85)
                if close is not None:
                    c_x, c_y = close
                    pyautogui.moveTo(c_x + 10, c_y + 10, duration=0.5)
                    pyautogui.click()
                else:
                    keyboard.press_and_release('ctrl+c')
                    time.sleep(0.5)
                    copied_text = pyperclip.paste()
                    if ''.join(copied_text.split()[:6]) not in my_response:
                        response = generate_response(copied_text)
                        my_response.append(''.join(response.split()[:6]))
                        reply_field = locate_button('img/post_reply.png', 0.7)
                        if reply_field is None:
                            for it in range(2):
                                pyautogui.scroll(-200)
                                time.sleep(0.2)
                        reply_field = locate_button('img/post_reply.png', 0.7)
                        if reply_field is None:
                            for it in range(2):
                                pyautogui.scroll(-200)
                                time.sleep(0.2)
                        time.sleep(2)
                        reply_field = locate_button('img/post_reply.png', 0.7)
                        pos_x, pos_y = reply_field
                        pyautogui.moveTo(pos_x + 30, pos_y + 10, duration=random.uniform(0.2, 1.5))
                        pyautogui.click()
                        pyautogui.write(response, interval=0.1)
                        time.sleep(random.uniform(0.4, 1.6))
                        reply_final = locate_button('img/reply_final.png', 0.7)
                        posr_x, posr_y = reply_final
                        pyautogui.moveTo(posr_x + 20, posr_y + 8, duration=random.uniform(0.2, 1.5))
                        pyautogui.click()
                    time.sleep(4)
                    home = locate_button('img/home.png', 0.7)
                    hom_x, hom_y = home
                    pyautogui.moveTo(hom_x + 20, hom_y + 8, duration=random.uniform(0.2, 1.5))
                    pyautogui.click()
                    time.sleep(2)
                    for sc in range(8):
                        pyautogui.scroll(200)
                        time.sleep(0.1)
                    time.sleep(60)
            else:
                amount_scroll = random.randrange(10, 15)
                for scroll in range(amount_scroll):
                    pyautogui.scroll(-200)
                    time.sleep(0.2)
                time.sleep(random.randrange(1, 2))
                for scroll_up in range(amount_scroll):
                    pyautogui.scroll(200)
                    time.sleep(0.2)
                print('no new tweets')
        except:
            time.sleep(4)
            home = locate_button('img/home.png', 0.7)
            hom_x, hom_y = home
            pyautogui.moveTo(hom_x + 20, hom_y + 8, duration=random.uniform(0.2, 1.5))
            pyautogui.click()
            time.sleep(2)
            for sc in range(8):
                pyautogui.scroll(200)
                time.sleep(0.1)

if __name__ == '__main__':
    overview_new_tweets()
