import re
import os
import openai
import pygame as pg


def generate_prompt(root_dir):
    code_input = ""
    for file in os.listdir(root_dir):
        if file.endswith(".py") and not file.startswith("dynamic_edit"):
            code = "\n# " + file + "\n"
            with open(os.path.join(root_dir, file), 'r') as f:
                code += f.read()
            code_input += code
    return code_input


class InputBox:
    def __init__(self, x, y, w, h, text='', color_inactive=pg.Color('lightskyblue3'),
                 color_active=pg.Color('dodgerblue2'), font=pg.font.Font(None, 32)):
        self.rect = pg.Rect(x, y, w, h)
        self.color = color_inactive
        self.text = text
        self.txt_surface = font.render(text, True, self.color)
        self.active = False
        self.color_inactive = color_inactive
        self.color_active = color_active
        self.font = font
        openai.organization = "org-JA36x78VZXZo26nnU7E2CNpc"
        os.environ["OPENAI_API_KEY"] = "insert api key here"
        k = os.getenv("OPENAI_API_KEY")
        openai.api_key = k
        self.root_dir = os.path.dirname(os.path.realpath(__file__))

    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = self.color_active if self.active else self.color_inactive
        if event.type == pg.KEYDOWN:
            if self.active:
                if event.key == pg.K_RETURN:
                    print(self.text)
                    self.generate_code(self.text)
                    self.text = ''
                elif event.key == pg.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = self.font.render(self.text, True, self.color)

    def update(self):
        # Resize the box if the text is too long.
        width = max(200, self.txt_surface.get_width()+10)
        self.rect.w = width

    def draw(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect.
        pg.draw.rect(screen, self.color, self.rect, 2)

    def edit_files(self, model_output):
        # split the files according to the pattern '\n# <file_name>.py'
        files = re.split('\n# [a-z]*.py\n', model_output)
        # read files and edit them
        idx = 1
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".py") and not filename.startswith("dynamic_edit"):
                self.edit_file(filename, files[idx])
                idx += 1

    def generate_code(self, text):
        code_input = generate_prompt(self.root_dir)
        response = openai.Edit.create(
            model="code-davinci-edit-001",
            input=code_input,
            instruction=text,
            temperature=0.0
        )
        self.edit_files(response['choices'][0]['text'])

    def edit_file(self, filename, model_output):
        # write new file
        with open(os.path.join(self.root_dir, "edited_" + filename), 'w') as f:
            f.writelines(model_output)



