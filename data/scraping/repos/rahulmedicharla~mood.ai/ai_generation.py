import openai, ast, threading
from queue import Queue

class Generation:
    def __init__(self, video, audio):
        self.openaikey = ""
        self.chat_prompt = """I will give you a bunch of lists. I want you to choose a random item from each of the lists and generate 5 unique prompts that can be used to create AI art. ONLY USE EACH ITEM ONCE. I want each prompt you give to follow a unique and fancy art style different from the others. The first list is a list of captions describing the video. Only use this value in TWO of the prompts generated. The second list is a list of objects. The third list if a list of emotions. the fourth list is a list of colors given in RGB. ONLY for this list you should convert the RGB values to their closest color name and select three colors to use. The fourth list is energy levels, and the fifth list is a list of sentiments. I would like at least one prompt to emphasize emotions through color, and at least one prompt be a fully abstracted version of the first list, and use warm colors for a high energy level and cool colors for a low energy level. Use random weights to determine how important a piece of information is for the prompt. Generate titles for each image as well. For some of the prompts follow art styles from famous artists. ONLY GIVE ME THE YOUR RESPONSE IN THE FORMAT ["title 1", "prompt 1", "title 2", "prompt 2", "titile 3", "prompt3"]

                            Follow these examples:
                            [['a man laying on a couch looking at his phone'], ['chair', 'couch', 'person', 'cellphone', 'bottle', 'tv'], ['happy', 'admiration', 'gratitude', 'neutral'], [(167, 184, 182), (145, 172, 173), (147, 173, 171), (142, 168, 166), (136, 162, 165), (141, 161, 158), (147, 159, 158), (145, 160, 158), (138, 150, 149)], ['High'], ['POSITIVE']]
                            your response: ["Connected", "Using a mixed media approach inspired by David Hockney, create an abstracted version of a man laying on a couch looking at his phone. Incorporate the colors RGB (167, 184, 182) and RGB (147, 173, 171) to emphasize a sense of connection and modern technology. Use a chair, cellphone, and tv in the composition. Digital art.", "Joyous Bubbles", "Using a warm color palette of reds, yellows, and oranges inspired by Kandinsky, create a playful scene featuring a person blowing bubbles. Incorporate the emotion happy to emphasize the joyful mood. Use a couch and bottle in the composition to add depth and interest. ","The Power of Gratitude ", "Using a cool color palette of blues and greens inspired by Monet, create a tranquil scene featuring a person sitting on a chair with a look of gratitude on their face. Incorporate the colors RGB(136, 162, 165) and RGB(141, 161, 158) to emphasize a peaceful mood. Use a bottle in the composition to add a subtle point of interest. Digital art.", "Neutral Haven ", "Using a futuristic approach, create an abstracted version of a man laying on a couch looking at his phone. Use the color RGB (142, 168, 166) to emphasize a sense of calm and neutrality. Incorporate a chair and tv in the composition to add interest.", "Basketball Dreams", "Using a mixed media approach inspired by Jean-Michel Basquiat, create a vibrant scene featuring a person holding a basketball. Use the color RGB (145, 160, 158) to emphasize energy and motion. Incorporate the emotion admiration to emphasize the subject's love for basketball. Digital art."]

                            [['a room with a television, a desk, and a chair '], ['person', 'dog', 'tv', 'book', 'chair', 'remote', 'couch', 'bed'], ['neutral', 'excited' 'angry', 'admiration'], [(229, 232, 230), (240, 198, 165), (239, 197, 164), (231, 233, 226), (231, 178, 146), (226, 231, 231), (228, 231, 229), (236, 238, 231)], ['Low'], ['POSITIVE', 'NEUTRAL']]
                            your response: ["Lost Memories", "Using a surrealist approach inspired by Salvador Dali, create an abstracted version of a person standing in a misty forest. Incorporate the caption 'a person walking through a forest' to add a sense of mystery. Use the colors RGB(95, 78, 63) and RGB(170, 160, 143) to emphasize a dreamlike atmosphere. Incorporate a book in the composition to add depth and interest. Digital art.", "Ethereal Garden", "Using a watercolor approach inspired by Claude Monet, create a tranquil scene featuring a flower garden. Use the colors RGB(214, 240, 190), RGB(164, 207, 63) and RGB(255, 208, 92) to emphasize a sense of serenity and tranquility. Incorporate the object 'chair' to add interest and depth. Incorporate the emotion 'admiration' to add a sense of awe.", "Desert Sunsets", "Using a digital approach inspired by James Turrell, create a futuristic version of a person sitting on a desert landscape during sunset. Use the colors RGB(247, 201, 133), RGB(199, 123, 72) and RGB(60, 27, 22) to emphasize a warm and peaceful atmosphere. Incorporate the object 'couch' to add interest and depth. Incorporate the emotion 'neutral' to add a sense of calmness.", "Deep Blue", "Using an expressionist approach inspired by Wassily Kandinsky, create an abstracted version of a person diving underwater. Use the colors RGB(38, 68, 99), RGB(89, 147, 171) and RGB(192, 217, 216) to emphasize the depth and fluidity of the underwater world. Incorporate the object 'camera' to add interest and depth. Incorporate the emotion 'excited' to add a sense of adventure. Digital art.", "Jungle Rhapsody", "Using a collage approach inspired by Henri Matisse, create a vibrant scene featuring a jungle. Use the colors RGB(0, 73, 83), RGB(255, 142, 83) and RGB(235, 219, 94) to emphasize a playful and lively atmosphere. Incorporate the objects 'person' and 'book' to add interest and depth. Incorporate the emotion 'happy' to add a sense of joy. Digital art."]
                            
                            REMEMBER TO INCLUDE FAMOUS ART STYLES IN YOUR PROMPT, MAKE THE PROMPTS AESTHICALLY PLEASING!
                        """
        
        self.chat_response = ""
        self.video_analysis = video
        self.audio_analysis = audio
        self.dalle_prompts = []
        self.image_results = []

    def generate_chat_prompts(self, openaikey):
        emotion_list = self.video_analysis.video_detected_emotions
        if 'sad' in emotion_list:
            emotion_list.remove('sad')

        openai.api_key = openaikey
        self.openaikey = openaikey

        message = str([self.video_analysis.video_classification,self.video_analysis.video_detected_objects, emotion_list, self.video_analysis.video_top_colors, self.audio_analysis.energy_level, self.audio_analysis.sentiment_analysis])

        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role" : "system", "content": self.chat_prompt},
                {"role" : "user", "content": message}
            ]
        )

        self.chat_response = completion.choices[0].message.content

        
    def generate_images(self, results, title, prompt):
        openai.api_key = self.openaikey

        image = openai.Image.create(
                prompt = prompt,
                n=1,
                size = "256x256"
            )
        
        results.put({
                "title": title,
                "link": image["data"][0]["url"],
            })
    
    
    def create_images(self):
        results = Queue()
        self.dalle_prompts = ast.literal_eval(self.chat_response)
        image_one_thread = threading.Thread(target=self.generate_images, args=(results, self.dalle_prompts[0], self.dalle_prompts[1]))
        image_two_thread = threading.Thread(target=self.generate_images, args=(results, self.dalle_prompts[2], self.dalle_prompts[3]))
        image_three_thread = threading.Thread(target=self.generate_images, args=(results, self.dalle_prompts[4], self.dalle_prompts[5]))
        image_four_thread = threading.Thread(target=self.generate_images, args=(results, self.dalle_prompts[6], self.dalle_prompts[7]))
        image_five_thread = threading.Thread(target=self.generate_images, args=(results, self.dalle_prompts[8], self.dalle_prompts[9]))

        image_one_thread.start()
        image_two_thread.start()
        image_three_thread.start()
        image_four_thread.start()
        image_five_thread.start()

        image_one_thread.join()
        image_two_thread.join()
        image_three_thread.join()
        image_four_thread.join()
        image_five_thread.join()

        while not results.empty():
            image = results.get()
            self.image_results.append(image)

        

            
