import pynecone as pc
import openai

openai.api_key = "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"

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

    def get_image(self):
        """Get the image from the prompt."""
        response = openai.Image.create(prompt=self.prompt, n=1, size="1024x1024")
        self.image_url = response["data"][0]["url"]
        self.image_processing = False
        self.image_made = True

def index():
    return pc.center(
        pc.vstack(
            pc.heading("DALL·E", font_size="1.5em"),
            pc.input(placeholder="Enter a prompt..", on_blur=State.set_prompt),
            pc.button(
                "Generate Image",
                on_click=[State.process_image, State.get_image],
                width="100%",
            ),
            pc.divider(),
            pc.cond(
                State.image_processing,
                pc.circular_progress(is_indeterminate=True),
                pc.cond(
                     State.image_made,
                     pc.image(
                         src=State.image_url,
                         height="25em",
                         width="25em",
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
app.add_page(index, title="Pynecone:DALL·E")
app.compile()
