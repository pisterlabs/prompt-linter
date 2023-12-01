import keyboard
import pyautogui
import pytesseract
import openai

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\alson\OneDrive\Documents\Script\tesseract.exe"
)
# Path to tesseract.exe example path: r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
openai.api_key = "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"


def click_button(button):
    # Coordinates for each button
    button_coords = {1: (415, 865), 2: (1415, 865), 3: (415, 1037), 4: (1415, 1055)}
    x, y = button_coords[button]
    pyautogui.click(x, y)


def chatGPT_answer(question_and_answers):
    # Adding explicit instruction to the prompt
    # Modify the instruction to the model to fit your needs
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in answering multiple-choice questions who only responds with the button number corresponding to the most likely answer do not respond with words only a integer. You'll do this even if the question involves content you can't analyze, such as videos or images. If you cannot answer the question, you'll respond with an educated guess. Remember only respond with an integer between 1 and 4 that corresponds to the answer.",
        },
        {"role": "user", "content": question_and_answers},
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=conversation
        )

        # Extract the assistant's reply
        reply = response["choices"][0]["message"]["content"].strip()
        # Return the button number as an integer
        return int(reply)
    except ValueError:
        # Handle exception here if the reply is not an integer
        print(f"Unexpected reply: {reply}")
        return None
    except:
        # Handle exception here if something else goes wrong
        print("Something went wrong.")
        print(response)
        return None


def kahoot_solver():
    # Coordinates for kahoot question and answers
    question_and_answers = {
        0: {"top_left": (200, 100), "bottom_right": (1800, 200)},
        1: {"top_left": (30, 800), "bottom_right": (940, 930)},
        2: {"top_left": (970, 800), "bottom_right": (1900, 930)},
        3: {"top_left": (30, 940), "bottom_right": (940, 1070)},
        4: {"top_left": (970, 940), "bottom_right": (1900, 1070)},
    }
    res = ""
    for element in question_and_answers:
        coords = question_and_answers[element]
        x1, y1 = coords["top_left"]
        x2, y2 = coords["bottom_right"]

        # Calculate width and height for each element for screenshot
        width = x2 - x1
        height = y2 - y1

        screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
        screenshot_text = pytesseract.image_to_string(screenshot)
        if not screenshot_text.strip():
            print("No text detected.")
            continue
        if element == 0:
            res += f"Question: {screenshot_text}"
        else:
            res += f"{element}: {screenshot_text}"
    print(res)
    try:
        answer = chatGPT_answer(res)
        click_button(answer)
        print(f"Answer: {answer}")
    except:
        print("chatGPT_answer failed")


# Bind the function to a hotkey
keyboard.add_hotkey("ctrl+alt+t", kahoot_solver)
# Keep the program running
keyboard.wait()
