import os
import openai
import urllib.request


def download_img(url, filename):
    save_path = 'downloads'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    urllib.request.urlretrieve(url, os.path.join(save_path, filename))


def main():
    image_desc = input("Image description: ")
    filename = input("Filename(*.png): ")
    openai.api_base = os.environ.get('OPENAI_API_BASE')
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    response = openai.Image.create(
        prompt=image_desc,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    download_img(image_url, filename)

if __name__ == "__main__":
    main()
