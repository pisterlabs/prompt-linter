import pygame
import openai
import sys

openai.api_key = 'insert_key'


def chat_with_chatgpt(user_calmness, keyword, model="gpt-3.5-turbo"):
    if user_calmness <= 3:
        calmness_description = "very agitated"
    elif 4 <= user_calmness <= 7:
        calmness_description = "moderately calm"
    else:
        calmness_description = "very calm"

    prompt = f"""
    You are an AI assistant helping in a unique scenario. Your task is to suggest commands for another voice assistant named Alessca, who is interacting with a paralyzed patient. The patient communicates with Alessca using an EEG-based thought recognition system. This system provides a single keyword which reflects the patient's thoughts. In addition, you have information about the patient's current state of calmness, which is described as '{calmness_description}'.

    For example, if the patient's calmness level is very low and the keyword is 'son', the commands might be like "Call the patient's son and inform him about the situation" or "Instruct the son to call 101". This indicates that the commands should be related to the keyword and sensitive to the patient's calmness level. 

    Similarly, if the calmness level is very low and the keyword is 'food', the commands might be like "Order food delivery for the patient" or "Call a nurse to bring food", as the patient is likely in need of food.

    On the other hand, if the calmness level is very high, for instance 9, and the keyword is 'son', the commands could be more casual and relaxed, like "Text my son to invite him for a walk in the sun."

    The current calmness level of the patient is: {user_calmness}
    The keyword from the patient's thoughts is: '{keyword}'

    Based on this information, you need to generate exactly four potential commands for Alessca. These commands should be short, concise, and in the format of an instruction that Alessca can act upon.

    Output your response as a Python list of exactly four short command strings no more than 8 words each command. Include no other text or output. Here is the specific format: ['command 1', 'command 2', 'command 3', 'command 4'].
    """

    response = openai.ChatCompletion.create(
        model = model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens = 100
    )

    message = response.choices[0].message['content']

    commands = message.replace('[', '').replace(']', '').replace("'", "").split(', ')

    return commands


class renderer():
    def __init__(self):
        pygame.init()
        self.is_first_call = True
        # TODO: change screen's dimensions
        self.screen_width = 1000
        self.screen_height = 750
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen_width = 800
        self.screen_height = 750
        pygame.display.set_caption("Menu Example")
        self.font = pygame.font.Font(None, 25)

        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)

        stress_threshold = 50
        self.stress_level = 0
        self.is_stressed = self.stress_level > stress_threshold

        self.option_names = ["Son", "Food", "Bathroom", "Emergency"]
        options = [
            [(0, 0, self.screen_width // 2, self.screen_height // 2), self.option_names[0], self.RED],
            [(self.screen_width // 2, 0, self.screen_width // 2, self.screen_height // 2), self.option_names[1], self.GREEN],
            [(0, self.screen_height // 2, self.screen_width // 2, self.screen_height // 2), self.option_names[2], self.BLUE],
            [(self.screen_width // 2, self.screen_height // 2, self.screen_width // 2, self.screen_height // 2), self.option_names[3], self.YELLOW]
        ]

        self.current_menu = options
        self.current_pos = 0
        self.is_ENTER = 0

    def render_gui(self, gaze_input, stress_level):
        self.stress_level = stress_level
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if gaze_input != 3:
            self.current_pos = (self.current_pos + gaze_input) % 4
        else:
            self.is_ENTER = 1 if gaze_input == 3 else 0

        if self.is_ENTER and self.is_first_call:
            if self.option_names[self.current_pos] == self.option_names[0]:
                new_options = chat_with_chatgpt(self.stress_level, self.option_names[self.current_pos])
                self.current_menu = [
                    [(0, 0, self.screen_width // 2, self.screen_height // 2), new_options[0], self.RED],
                    [(self.screen_width // 2, 0, self.screen_width // 2, self.screen_height // 2), new_options[1],
                     self.GREEN],
                    [(0, self.screen_height // 2, self.screen_width // 2, self.screen_height // 2),
                     new_options[2], self.BLUE],
                    [(self.screen_width // 2, self.screen_height // 2, self.screen_width // 2, self.screen_height // 2),
                     new_options[3], self.YELLOW]
                ]
                self.is_first_call = False
            elif self.option_names[self.current_pos] == self.option_names[1]:
                new_options = chat_with_chatgpt(self.stress_level, self.option_names[self.current_pos])
                self.current_menu = [
                    [(0, 0, self.screen_width // 2, self.screen_height // 2), new_options[0], self.RED],
                    [(self.screen_width // 2, 0, self.screen_width // 2, self.screen_height // 2), new_options[1],
                     self.GREEN],
                    [(0, self.screen_height // 2, self.screen_width // 2, self.screen_height // 2),
                     new_options[2], self.BLUE],
                    [(self.screen_width // 2, self.screen_height // 2, self.screen_width // 2, self.screen_height // 2),
                     new_options[3], self.YELLOW]
                ]
                self.is_first_call = False
            elif self.option_names[self.current_pos] == self.option_names[2]:
                new_options = chat_with_chatgpt(self.stress_level, self.option_names[self.current_pos])
                self.current_menu = [
                    [(0, 0, self.screen_width // 2, self.screen_height // 2), new_options[0], self.RED],
                    [(self.screen_width // 2, 0, self.screen_width // 2, self.screen_height // 2), new_options[1],
                     self.GREEN],
                    [(0, self.screen_height // 2, self.screen_width // 2, self.screen_height // 2),
                     new_options[2], self.BLUE],
                    [(self.screen_width // 2, self.screen_height // 2, self.screen_width // 2, self.screen_height // 2),
                     new_options[3], self.YELLOW]
                ]
                self.is_first_call = False
            elif self.option_names[self.current_pos] == self.option_names[3]:
                new_options = chat_with_chatgpt(self.stress_level, self.option_names[self.current_pos])
                self.current_menu = [
                    [(0, 0, self.screen_width // 2, self.screen_height // 2), new_options[0], self.RED],
                    [(self.screen_width // 2, 0, self.screen_width // 2, self.screen_height // 2), new_options[1],
                     self.GREEN],
                    [(0, self.screen_height // 2, self.screen_width // 2, self.screen_height // 2),
                     new_options[2], self.BLUE],
                    [(self.screen_width // 2, self.screen_height // 2, self.screen_width // 2, self.screen_height // 2),
                     new_options[3], self.YELLOW]
                ]
                self.is_first_call = False

        self.screen.fill(self.GRAY)

        for i, (rect, text, color) in enumerate(self.current_menu):
                pygame.draw.rect(self.screen, color, rect)
                if i == self.current_pos:
                    if self.is_first_call:
                        pygame.draw.rect(self.screen, self.BLACK, rect, 30)
                    else:
                        pygame.draw.rect(self.screen, self.WHITE, rect, 30)
                text_surface = self.font.render(text, True, self.BLACK)
                text_rect = text_surface.get_rect(center=(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2))
                self.screen.blit(text_surface, text_rect)

        scale_rect = pygame.Rect(850, 0, 100, self.screen_height)
        pygame.draw.rect(self.screen, ((255 * self.stress_level) / 10, (255 * (100 - self.stress_level*10)) / 100, 0), scale_rect)

        arrow_height = int((self.stress_level / 10) * self.screen_height)
        arrow_pos = (self.screen_width - 50, self.screen_height - arrow_height)

        arrow_top = max(scale_rect.top, arrow_pos[1] - 15)  # Ensure the arrow stays within the scale
        arrow_bottom = min(scale_rect.bottom, arrow_pos[1] + 15)

        pygame.draw.polygon(self.screen, self.BLACK,
                            [(1000, arrow_bottom), (self.screen_width, arrow_pos[1]), (1000, arrow_top)])

        pygame.display.flip()