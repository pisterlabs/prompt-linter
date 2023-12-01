import pygame
import pygame_gui
from collections import deque
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import BaseCallbackHandler
import re


pygame.init()


DISPLAY_WIDTH = 1500  
DISPLAY_HEIGHT = 800
CLOSE_TO_BORDER_DISTANCE = 10


screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
clock = pygame.time.Clock()


manager = pygame_gui.UIManager((DISPLAY_WIDTH, DISPLAY_HEIGHT))

# Creating input box and button
input_box = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((50, 700), (100, 40)), manager=manager)
button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((200, 700), (90, 30)), text='Use last text', manager=manager)


last_text = ''
past_decisions = []
past_text = deque(maxlen=5)

# Custom handler for langchain
class MyCustomHandler(BaseCallbackHandler):  
    def on_llm_new_token(self, token: str, **kwargs) -> None:  
        pass

callback_manager = CallbackManager(handlers=[MyCustomHandler])

longllm = LlamaCpp(
    model_path="/YourPath", 
    callback_manager=callback_manager, 
    max_tokens=20,
    temperature=1,
    top_p=0.8,
    repeat_penalty=1.3,
    top_k=10,
    last_n_tokens_size=32,
    n_batch = 500,
    verbose=False,
    use_mlock= True,
    n_threads=5,
)

# Decision validation function
def validate_decision(decision):
    # Define a regular expression pattern for the decision
    pattern = r'\((\d{1,4}),(\d{1,4})\)\s*\((\d{1,3}),(\d{1,3}),(\d{1,3})\)'
    match = re.search(pattern, decision.strip())

    # If the decision matches the pattern, extract the x, y, and color values
    if match:
        past_decisions.append(decision)
        x, y, r, g, b = map(int, match.groups())

        # Check if the position is within the screen boundaries and the color values are valid
        if 0 <= x <= DISPLAY_WIDTH and 0 <= y <= DISPLAY_HEIGHT and all(0 <= c <= 255 for c in [r, g, b]):
            return (x, y), (r, g, b)
    # If the decision does not match the pattern or the position is outside the screen, return None
    return None

# Function to execute decision
def execute_decision(decision, text):
    result = validate_decision(decision)
    
    if result != None:
        (x, y), color = result
        new_square = square(x, y, color, squares, text)
        squares.add(new_square) 
        return True  # return True if a square is drawn
    return False  # return False if no square is drawn

# square class
class square(pygame.sprite.Sprite):
    def __init__(self,x,y, color,group, prompt):
        super().__init__(group) 
        self.image = pygame.Surface((15,15))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.prompt = prompt
        self.font = pygame.font.SysFont("Verdana", 7)
        self.text = self.font.render(self.prompt, True, (165, 42, 42))
        self.bg = pygame.Rect(x, y, 15, 15)
        self.pos = self.text.get_rect(center = (self.bg.x + self.bg.width/2,(self.bg.y + self.bg.height/2)))
        
    def test(self,display):
        self.pos = self.text.get_rect(center = (self.bg.x + self.bg.width/2,
                                            self.bg.y+20 + self.bg.height/2))
        display.blit(self.text, self.pos)

# Creating square group
squares = pygame.sprite.Group()

# Creating decision template
decision_template = (
    """As a language model, your task is to guide the drawing process with words. 
    Instead of responding to the following text in a conventional manner, 
    your response should be a position on the screen in the form of (X,Y) and a color in the form of (R,G,B). 
    The text you are interpreting is: '{question}'. 
    Choose X and Y as integers between 0 and {DISPLAY_WIDTH}and 0 and {DISPLAY_HEIGHT} respectively. 
    Choose R, G, and B as integers between 0 and 255.
    Your response should be in the format: (X,Y) (R,G,B). 
    Remember, your role is not to answer the question but to guide the drawing process. 
    Now, proceed with the guidance.\n\n###"""
)
prompt = PromptTemplate(template=decision_template, input_variables=["question", "DISPLAY_WIDTH", "DISPLAY_HEIGHT"])
llm_chain_decision = LLMChain(prompt=prompt, llm=longllm)

# Main game loop
draw_square = False
square_drawn = False
start = False
while True:
    time_delta = clock.tick(60)/1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        manager.process_events(event)

        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == button:
                    start = True
                    text = last_text
                    draw_square = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                start = True
                text = input_box.get_text()
                draw_square = True

    manager.update(time_delta)

    screen.fill((255, 255, 255))

    if start and draw_square and not square_drawn:
        text = input_box.get_text()
        last_text = text

        decision = llm_chain_decision.predict(question=text,DISPLAY_WIDTH = DISPLAY_WIDTH, DISPLAY_HEIGHT = DISPLAY_HEIGHT)
        print(decision)
        square_drawn = execute_decision(decision, text)

    if square_drawn:
        square_drawn = False
        draw_square = False

    for c in squares:
        c.test(screen)
    squares.draw(screen)
    manager.draw_ui(screen)
    pygame.display.update()
