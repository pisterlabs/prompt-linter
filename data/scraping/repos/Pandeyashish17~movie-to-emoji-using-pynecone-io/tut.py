import pynecone as pc
import openai

openai.api_key = "YOUR_API_KEY"

class State(pc.State):
    """The app state."""
    prompt = ""
    image_url = ""
    image_processing = False
    image_made = False

    def process_image(self):
        """Set the image processing flag to true and indicate image is not made yet."""
        self.image_processing = True
        self.image_made = False        

    def generate_emoji(self):
        """Get the emojis from the prompt."""
        response = openai.Completion.create(engine="text-davinci-003", prompt=f"{self.prompt} emojis", max_tokens=30)
        self.image_url = response["choices"][0]["text"]
        self.image_processing = False
        self.image_made = True

def index():
    return pc.center(
        pc.vstack(
            pc.heading("Movie to Emoji Converter", font_size="1.5em"),
            pc.input(placeholder="Enter a movie title..", on_blur=State.set_prompt),
            pc.button(
                "Generate Emoji",
                on_click=[State.process_image, State.generate_emoji],
                width="100%",
            ),
            pc.divider(),
            pc.cond(
                State.image_processing,
                pc.circular_progress(is_indeterminate=True),
                pc.cond(
                     State.image_made,
                     pc.text(
                         State.image_url
                    )
                )
            ),
            bg="white",
            padding="2em",
            shadow="lg",
            border_radius="lg",
        ),
        width="100%",
        height="100vh",
        bg="radial-gradient(circle at 22% 11%,rgba(62, 180, 137,.20),hsla(0,0%,100%,0) 19%)",
    )

# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="Movie To Emoji")
app.compile()