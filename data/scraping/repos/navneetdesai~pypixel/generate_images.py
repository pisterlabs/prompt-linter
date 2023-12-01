from pypixelai import PyPixel
from pypixelai.models import OpenAI


def main():
    model = OpenAI()  # choose a model
    px = PyPixel(model, retries=3)  # initialize PyPixel with the model
    urls = px.generate_images("Blank white image", num_images=2, download=True)
    print(urls)


if __name__ == "__main__":
    main()
