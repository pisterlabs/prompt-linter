import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
image = openai.Image.create(
  prompt="Little labrador, background blue sky and grass ground, daylight, polaroid",
  n=2,
  size="1024x1024"
)

print(image['data'][0].url)
print(image)

# wygeneruj obraz stosując prompt z ćwiczeń Midjourney
# Porównaj wygenerowane obrazy
# Określ ilość tokenów
# Określ miesięczny koszt przy założeniu 1000 obrazów dziennie
