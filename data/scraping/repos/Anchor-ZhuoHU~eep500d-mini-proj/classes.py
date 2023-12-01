import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import spacy
import pickle
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity

import requests
import io
import cv2
import base64
from IPython.display import display
from openai import OpenAI
client = OpenAI()

from dijkstar import Graph

class EmbeddedImageFinder(object):
    text_file = "flower_species.txt"
    
    def __init__(self):
        with open(self.text_file, "r") as f:
            flowers = f.read().splitlines()
        self.flowers = flowers
        
        images = []
        for flower in flowers:
            path = "images/" + flower + ".png"
            image = np.asarray(Image.open(os.path.join(path)))
            images.append(image)
            
        images = [np.array(image).flatten() for image in images]
        self.images = images
        
        self.embeddings = self.run_embed()
        
    def run_embed(self):
        if os.path.exists("embeddings.pkl"):
            with open("embeddings.pkl", "rb") as pickle_file:
                embeddings_dict = pickle.load(pickle_file)
            return embeddings_dict
        else:
            embeddings_dict = {}        
            images = self.images
            for i, img in tqdm(enumerate(images)):
                embedding = img.reshape(1, -1)
                embeddings_dict[str(i)] = embedding
            with open(os.getcwd() + "/embeddings.pkl", 'wb') as pickle_file:
                pickle.dump(embeddings_dict, pickle_file)
                
            return self.run_embed()

    def find_most_similar(self, image):
        embeddings = self.embeddings
        image_reshaped = image.reshape(1, -1)
        
        max_sim = 0
        max_index = 0
        for i in range(100):
            sim = cosine_similarity(image_reshaped, embeddings[str(i)])[0][0]
            if sim > max_sim:
                max_sim = sim
                max_index = i
        # Return the most similar image and its index
        path = "images/" + self.flowers[max_index] + ".png"
        return path, max_index
    
    def find_most_similar_k(self, image, top_n=3):
        embeddings = self.embeddings
        image_reshaped = image.reshape(1, -1)
        
        sim = [(i, cosine_similarity(image_reshaped, embeddings[str(i)])[0][0])
                for i in range(100)]
        sim.sort(key=lambda x: x[1], reverse=True)
        nearest_index = [i for i, similarity in sim[1:top_n+1]]

        paths = []
        for j in range(top_n):
            paths.append("images/" + self.flowers[nearest_index[j]] + ".png")
        return paths, nearest_index

    def finder(self, upload):
        # Load another image from a file and store it in a variable
        another_image = np.asarray(Image.open(upload)) # change this to your file path
        another_image = resize(another_image, (1024, 1024))
        # Find the most similar image to the another image
        most_similar_image, most_similar_index = self.find_most_similar(another_image)

        # Display the most similar image and its index
        images_list = self.flowers[most_similar_index]
        return [
            "The most similar image is image " + str(images_list), 
            most_similar_image
        ]

class ChineseWhispers(object):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": "Bearer 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"}
    file_path = "chinese_whispers/Karthik.png"
    
    def __init__(self, save=True):
        self.save = save
        self.loop_list = self.loop_list_generator("r")
        i = int(self.loop_list[0])
        
        if i != 0:
            file_path = os.path.join(f"chinese_whispers/loops/{i}.png")
            self.file_path = file_path
        
        with open(self.file_path, "rb") as f:
            data = f.read()
        self.current_img = data
        self.current_text = ""
    
    def img2text(self):
        response = requests.post(self.API_URL, headers=self.headers, data=self.current_img)
        try: 
            return response.json()[0]["generated_text"]
        except:
            self.img2text(self)
    
    def text2img(self, text, i):
        # Generate an image from the flower name using DALL·E 2
        response = client.images.generate(
            model="dall-e-2",
            prompt=text,
            size="1024x1024",
            quality="standard",
            response_format="b64_json",
            n=1
        )
        # Get the image data from the response
        image_data = base64.b64decode(response.data[0].b64_json)
        # Create an image from the decoded data
        image = Image.open(io.BytesIO(image_data))
        display(image)
        if self.save == True:
            file_path = self.file_path.split("/")[0] + "/loops"
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            image_path = os.path.join(file_path, f"{i}.png")
            image.save(image_path)
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        byte_im = buf.getvalue()
        self.current_img = byte_im
        
    def loop_list_generator(self, mul):
        file_path = self.file_path.split("/")[0]
        if mul == "r":
            with open(file_path + "/loop_list.txt", "r") as f:
                loop_list = f.read().splitlines()
            return loop_list
        elif mul == "w":
            lines = self.loop_list
            with open(file_path + "/loop_list.txt", "w") as f:
                for line in lines:
                    f.write(f"{line}\n")
            
        
    def game_loop(self, n):
        for line in self.loop_list:
            i = int(line)
            try: 
                print(f"Loop_{i+1}: Now, from image to text...")
                self.current_text = self.img2text()
                print(f"Loop_{i+1} Text: {self.current_text}")
                
                print(f"Loop_{i+1}: Now, from text to image...")
                print(f"Loop_{i+1} Image: \n")
                self.text2img(self.current_text, i+1)

                self.loop_list = self.loop_list[1:]
            except:
                self.loop_list_generator("w")
                raise RuntimeError("Stuck! Need to run again! ")
              
def factory_chinese_whispers(file_path=None, loops=1000, save=True):
    new_chinese_whispers = ChineseWhispers(save)
    return new_chinese_whispers.game_loop(loops)

class SaveVideo(object):

    def __init__(self, text_path, video_name):
        self.images = np.loadtxt(text_path, dtype=str)
        self.video_name = video_name
        self.video_out_path = "video/" + video_name + '.mp4'

    def output_video(self):
        print(f'Saving video at {self.video_out_path}')
        result = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*'FMP4'), 20, (1024,1024))

        for frame, img_path in enumerate(self.images, 1):
            if frame%100 == 0:
                print('processing frame: ',frame)

            im_out = cv2.imread(img_path)
            cv2.putText(im_out,str(frame),(30,100),cv2.FONT_HERSHEY_PLAIN, 3 ,(0,0,255),thickness = 3)
            result.write(im_out)

        result.release()

def write_loops_txt(
        path="chinese_whispers/loops/",
        file_path="video",
        file_name="loops",
        n=100
):
    imgs = []
    for i in range(1, n+1):
        imgs.append(path + f"{i}" + '.png')

    with open(file_path + "/" + file_name + ".txt", "w") as f:
        for img in imgs:
            f.write(f"{img}\n")

class tools(object):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        with open("flower_species.txt", "r") as f:
            flowers = f.read().splitlines()
        self.names = []
        for flower in flowers:
                self.names.append(flower)
        
    def saveDict(self, dictionary, filename):
        f = open('{}.txt'.format(filename),'w')
        f.write(str(dictionary))
        f.close()
    def ReadDict(self, filename):
        f = open('{}.txt'.format(filename),'r')
        a = f.read()
        dictionary = eval(a)
        f.close()
        return dictionary
    def similar_word(self, input_word, top_n=3):
        
        nlp = self.nlp
        word_list = self.names
        input_vector = nlp(input_word).vector

        # Calculate cosine similarities
        similarities = [(word, np.dot(input_vector, nlp(word).vector) / (np.linalg.norm(input_vector) * np.linalg.norm(nlp(word).vector)))
                        for word in word_list]

        # Sort words by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get the nearest words
        nearest_words = [word for word, similarity in similarities[:top_n]]

        return nearest_words
    
def compute_dis_matrix(images):
    similarity_matrix = cosine_similarity(images)
    distance_matrix = 1 / (similarity_matrix + 2)
    return distance_matrix

def construct_map(flowers_name, distance_matrix):
  graph = Graph()
  new_finder = EmbeddedImageFinder()
  for i in tqdm(range(0, len(flowers_name))):
      path = "images/" + flowers_name[i] + ".png"
      image = np.asarray(Image.open(os.path.join(path)))
      paths, index = new_finder.find_most_similar_k(image, top_n = 10)
      for j in range(len(index)):
        graph.add_edge(flowers_name[i], flowers_name[index[j]], distance_matrix[i][index[j]])
        graph.add_edge(flowers_name[index[j]], flowers_name[i], distance_matrix[i][index[j]])
  return graph