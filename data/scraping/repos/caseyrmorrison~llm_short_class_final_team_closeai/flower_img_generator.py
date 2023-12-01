from openai import OpenAI
from PIL import Image
import urllib.request
from io import BytesIO
from IPython.display import display
import os
import time

# Get OpenAI key
open_ai_key_file = "api-key.txt"
with open(open_ai_key_file, "r") as f:
  for line in f:
    OPENAI_KEY = line
    break

# openai.api_key = OPENAI_KEY
client = OpenAI(api_key=OPENAI_KEY)

# Came accross an error where a species name was marked as against their policy - Cockscomb so I had to swap that for Cosmos and rerun this
## Run attempt 1 - error due to bad word
# list_of_flowers = ["Rose", "Tulip", "Orchid", "Lily", "Daffodil", "Sunflower", "Dahlia", "Iris", "Marigold", "Geranium", "Hyacinth", "Peony", "Chrysanthemum", "Lavender", "Begonia", "Carnation", "Azalea", "Snapdragon", "Gardenia", "Amaryllis", "Anemone", "Camellia", "Freesia", "Gladiolus", "Hibiscus", "Jasmine", "Lilac", "Lotus", "Magnolia", "Poppy", "Ranunculus", "Sweet pea", "Violet", "Zinnia", "Bleeding Heart", "Cherry Blossom", "Cockscomb", "Foxglove", "Heather", "Hollyhock", "Nasturtium", "Pansy", "Periwinkle", "Phlox", "Plumeria", "Primrose", "Rhododendron", "Scabiosa", "Thistle", "Wisteria", "Bluebell", "Borage", "Calendula", "Calla Lily", "Candytuft", "Columbine", "Cornflower", "Crocus", "Cyclamen", "Delphinium", "Forget-me-not", "Forsythia", "Fuchsia", "Garden Phlox", "Gypsophila", "Hellebore", "Hydrangea", "Ice Plant", "Impatiens", "Joe-Pye Weed", "Lantana", "Larkspur", "Lobelia", "Lupine", "Mimosa", "Osteospermum", "Petunia", "Protea", "Queen Anne's Lace", "Rudbeckia", "Salvia", "Statice", "Tansy", "Trillium", "Verbena", "Witch Hazel", "Yarrow", "Agapanthus", "Alstroemeria", "Aster", "Bellflower", "Blanket Flower", "Butterfly Bush", "Coreopsis", "Dianthus", "Echinacea", "Gaillardia", "Gerbera Daisy", "Honeysuckle", "Morning Glory"]

## Run attempt 2 with the rest of the list worked but produced some images that weren't flowers
# list_of_flowers = ["Cosmos", "Foxglove", "Heather", "Hollyhock", "Nasturtium", "Pansy", "Periwinkle", "Phlox", "Plumeria", "Primrose", "Rhododendron", "Scabiosa", "Thistle", "Wisteria", "Bluebell", "Borage", "Calendula", "Calla Lily", "Candytuft", "Columbine", "Cornflower", "Crocus", "Cyclamen", "Delphinium", "Forget-me-not", "Forsythia", "Fuchsia", "Garden Phlox", "Gypsophila", "Hellebore", "Hydrangea", "Ice Plant", "Impatiens", "Joe-Pye Weed", "Lantana", "Larkspur", "Lobelia", "Lupine", "Mimosa", "Osteospermum", "Petunia", "Protea", "Queen Anne's Lace", "Rudbeckia", "Salvia", "Statice", "Tansy", "Trillium", "Verbena", "Witch Hazel", "Yarrow", "Agapanthus", "Alstroemeria", "Aster", "Bellflower", "Blanket Flower", "Butterfly Bush", "Coreopsis", "Dianthus", "Echinacea", "Gaillardia", "Gerbera Daisy", "Honeysuckle", "Morning Glory"]

## Run attempt 3 with more detailed description of flowers that produced non flower images
list_of_flowers = ['hellebore flower species', 'delphinium flower species', 'candytuft flower species']

# Store the URLs generated for each photo
url_list = []
batch_size = 1

# Iterate through the list of flowers and call the API
for x in list_of_flowers:
  print(x)
  response = client.images.generate(
      model="dall-e-2",
      prompt=x,
      size="256x256",
      quality="standard",
      n=batch_size,
      style="natural"
  )
  url_list.extend([obj.url for obj in response.data])
  print(url_list)

  # Open the image URL
  image_url = response.data[0].url
  with urllib.request.urlopen(image_url) as url:
    image = Image.open(BytesIO(url.read()))

  # Save the file into a google drive folder
  img_path = '/content/drive/MyDrive/openai/img/' + x + '.jpg'
  image.save(img_path)
  # display(image)

  # Wait due to rate limiting for OpenAI API calls only 5 calls per 1 minute
  time.sleep(15)