from pypixel import PyPixel
from pypixel.models import OpenAI


def main():
    model = OpenAI()  # choose a model
    px = PyPixel(model, retries=3)  # initialize PyPixel with the model
    code = px.edit_images(
        image=open("image.png", "rb"),
        mask=open("mask.png", "rb"),
        prompt="A sunlit indoor lounge area with a pool containing a flamingo",
        n=1,
        size="1024x1024",
        download=True,
    )
    print(code)


if __name__ == "__main__":
    main()
