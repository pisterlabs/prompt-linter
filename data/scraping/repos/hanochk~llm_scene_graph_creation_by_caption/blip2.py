import os
import torch
#import nltk
from PIL import Image
import openai
from arango import ArangoClient
import requests # request img from web
import shutil # save img locally
from lavis.models import load_model_and_preprocess

dbname = "ipc_200"
arango_host = "http://172.83.9.249:8529"

class BLIP2():
    def __init__(self, model="blip2_t5", question_model=False):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.question_model = question_model
        if self.question_model:

            if model=="blip2_t5":
                model_type = "pretrain_flant5xxl"
            elif model == "blip2":
                model_type = "pretrain"
            else:
                raise
            self.question_model, self.question_vis_processors, question_text_processors = load_model_and_preprocess(name=model, model_type=model_type, is_eval=True, device=self.device)

        self.match_model, self.match_vis_processors, self.match_text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=self.device, is_eval=True)

    def process_image_and_captions(self, file, caption, file_or_url='file', match_head="itc", verbose=False): #https://github.com/salesforce/LAVIS/blob/3446bac20c5646d35ae383ebe6d13cec4f8b00cb/examples/blip2_image_text_matching.ipynb

        if file_or_url == 'url':
            file = self.download_file(file) 
        # load sample image
        raw_image = Image.open(file).convert("RGB")
        #raw_image.crop()
        #display(raw_image.resize((596, 437)))

        img = self.match_vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        txt = self.match_text_processors["eval"](caption)        
        it_score = self.match_model({"image": img, "text_input": txt}, match_head=match_head)
        if verbose and  match_head == "itc":
            print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)

        if match_head == "itm":
            it_score = torch.nn.functional.softmax(it_score, dim=1)[0][:1]
        if verbose:
            print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')        

        return it_score
        
class BLIP2GPTDialog():
    def __init__(self, blip, db_name=None, db_host=None):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        if db_name and db_host:
            client = ArangoClient(hosts=db_host)
            self.db = client.db(db_name, username='nebula', password='nebula')
        self.prompt_step1 = """The text below is an inaccurate image caption. 
        Ask 15 different clarifying questions to get more accurate information that can be seen in the image:
        
        """
        self.prompt_step1_1 = """The text below is an image caption. 
        Ask 5 different clarifying questions to get more details that can be seen in the image:
        
        """
        #self.prompt_step2 = "Text below With the provided weak and inaccurate captions for the image, taking into account clarifying questions and answers to them, write more accurate detailed captions for the same image: \n\n"
        self.prompt_step2 = """The following text contains weak descriptions of the image, which may contain incorrect, inaccurate or irrelevant information. 
        Taking in account clarifying questions and answers about this image, provide an accurate and detailed captions of it: 
        
        """
        if blip.question_model:
            self.question_model =  blip.question_model
            self.question_vis_processors = blip.question_vis_processors
        self.match_model = blip.match_model
        self.match_vis_processors = blip.match_vis_processors
        self.match_text_processors = blip.match_text_processors
    
#     def sumy_sum(self, captions, count):
#         LANGUAGE = "english"
#         SENTENCES_COUNT = int(count)
#         prompt = []
#         parser = PlaintextParser.from_string(captions, Tokenizer(LANGUAGE))    
#         stemmer = Stemmer(LANGUAGE)

#         summarizer = Summarizer(stemmer)
#         summarizer.stop_words = get_stop_words(LANGUAGE)

#         for sentence in summarizer(parser.document, SENTENCES_COUNT):
#             prompt.append(str(sentence))
#         return(prompt)

#     def get_mdfs(self, movie_id):
#         mdfs = []
#         for res in self.db.collection("s4_llm_output").find({'movie_id': movie_id}):
#             mdfs.append(res)
#         newlist = sorted(mdfs, key=lambda d: d['frame_num'])
#         return(newlist)
    
#     def get_persons_bboxes(self, movie_id, frame_num):
#         persons = []
#         for res in self.db.collection("s4_visual_clues").find({'movie_id': movie_id, 'frame_num': frame_num}):
#             for roi in res['roi']:
#                 if roi['bbox_object'] == 'person' and float(roi['bbox_confidence']) > 0.8:
#                     rois = []
#                     rois_str = roi['bbox'].replace("\"", "").replace("[", "").replace("]","").split(",")
#                     for r in rois_str:
#                         rois.append(float(r))
#                     #print("DEBUG ", rois)
#                     persons.append(rois)
#         return(persons)
#         #s4_visual_clues

#     def get_objects_bboxes(self, movie_id, frame_num):
#         persons = []
#         for res in self.db.collection("s4_visual_clues").find({'movie_id': movie_id, 'frame_num': frame_num}):
#             for roi in res['roi']:
#                 if roi['bbox_object'] != 'person' and float(roi['bbox_confidence']) > 0.5:
#                     rois = []
#                     rois_str = roi['bbox'].replace("\"", "").replace("[", "").replace("]","").split(",")
#                     for r in rois_str:
#                         rois.append(float(r))
#                     #print("DEBUG ", rois)
#                     persons.append((rois, roi['bbox_object']))
#         return(persons)

#     def download_file(self, url):
#         file_name = "/tmp/" + url.split("/")[-1] #prompt user for file_name
#         #print(file_name)
#         res = requests.get(url, stream = True)

#         if res.status_code == 200:
#             with open(file_name,'wb') as f:
#                 shutil.copyfileobj(res.raw, f)
#             print('Image sucessfully Downloaded: ',file_name)
#         else:
#             print('Image Couldn\'t be retrieved')
#         return(file_name)
    
#     def ask_gpt(self, prompt, text):
#         prompt = prompt + "\n"
#         prompt = prompt + text
#         try:
#             response = openai.Completion.create(
#             engine="text-davinci-003",
#             prompt=prompt,
#             temperature=0.0,
#             max_tokens=256,
#             #top_p=1.0,
#             frequency_penalty=0.0,
#             presence_penalty=0.0
#             )
#             #print(response['choices'][0]['text'])
#             return(response['choices'][0]['text'])
#         except Exception as e:
#             print(e)
#             return("No answer from OpenAI, please re-try in few minutes")

#     def get_visual_properities(self, raw_image, movie_id, frame_num):
#         bboxes = self.get_persons_bboxes(movie_id, frame_num)
#         counter_ = {0: 'first', 1:'second', 2:'third', 3:'fourth', 4:'fifth', 5:'sixth', 6:'seventh', 7:'eighth', 8:'ninth', 9: 'tenth'}
#         print("Persons props ========================================")
#         question_answer = ""
#         if len(bboxes) > 0:
#             for i, person in enumerate(bboxes):
#                 #print(person)
#                 object_image = raw_image.crop(person)
#                 #display(object_image.resize((596, 437)))
#                 question_image = self.question_vis_processors["eval"](object_image).unsqueeze(0).to(self.device)
#                 prompt = "Qestion: What kind of clothes is the person in the picture wearing? Answer:"
#                 question = "Qestion: What kind of clothes is the {} person in the picture wearing? Answer:".format(counter_[i])
#                 answer =  self.question_model.generate({"image": question_image, "prompt": prompt})
#                 question_answer = question_answer + question + " " + answer[0].split(",")[0] + "\n"
#                 #print(answer[0].split(",")[0])
#                 #----------------------------------------------------------------------
#                 prompt = "Qestion: What is a gender of person in the picture? Answer:"
#                 answer =  self.question_model.generate({"image": question_image, "prompt": prompt})
#                 question = "Qestion: What is a gender of {} person in the picture? Answer:".format(counter_[i])
#                 question_answer = question_answer + question + " " + answer[0].split(",")[0] + "\n"
#                 #print(answer[0].split(",")[0])
#                 #-----------------------------------------------------------------------
#                 prompt = "Qestion: How old is the person in the picture? Answer:"
#                 answer =  self.question_model.generate({"image": question_image, "prompt": prompt})
#                 question = "Qestion: How old is the {} person in the picture? Answer:".format(counter_[i])
#                 question_answer = question_answer + question + " " + answer[0].split(",")[0] + "\n"
#                 #print(answer[0].split(",")[0])
#                 #------------------------------------------------------------------------
#                 prompt = "Qestion: What is the physique of the person? Answer:"
#                 answer =  self.question_model.generate({"image": question_image, "prompt": prompt})
#                 question = "Qestion: What is the physique of the {} person? Answer:".format(counter_[i])
#                 question_answer = question_answer + question + " " + answer[0].split(",")[0] + "\n"

#                 #------------------------------------------------------------------------
#                 # prompt = "Qestion: What is the nationality of the person in the image? Answer:"
#                 # answer =  self.question_model.generate({"image": question_image, "prompt": prompt})
#                 # question = "Qestion: What is the nationality of the {} person in the image? Answer:".format(counter_[i])
#                 # question_answer = question_answer + question + " " + answer[0].split(",")[0] + "\n"

#                 #--------------------------------------------------------------------------
#                 prompt = "Qestion: What is the mood of the person in the image? Answer:"
#                 answer =  self.question_model.generate({"image": question_image, "prompt": prompt})
#                 question = "Qestion: What is the mood of the {} person in the image? Answer:".format(counter_[i])
#                 question_answer = question_answer + question + " " + answer[0].split(",")[0] + "\n"
#                 #print(answer[0].split(",")[0])
#         # bboxes = self.get_objects_bboxes(movie_id, frame_num)
#         # print("Objects props ========================================")
#         # if len(bboxes) > 0:
#         #     for obj in bboxes:
#         #         bbox = obj[0]
#         #         object_ = obj[1]
#         #         print("DEBUG ",object_)
#         #         object_image = raw_image.crop(bbox)
#         #         display(object_image.resize((596, 437)))
#         #         question_image = self.question_vis_processors["eval"](object_image).unsqueeze(0).to(self.device)
#         #         prompt = "Qestion: What is a color of the {} in the image? Answer:".format(object_)
#         #         answer =  self.question_model.generate({"image": question_image, "prompt": prompt})
#         #         question_answer.append(prompt + " " + answer[0].split(",")[0])
#         #         print(answer[0].split(",")[0])
#         #         prompt = "Qestion: How is a gender of person in the picture? Answer:"
#         #         answer =  self.question_model.generate({"image": question_image, "prompt": prompt})
#         #         print(answer[0])
#         #         prompt = "Qestion: How old is the person in the picture? Answer:"
#         #         answer =  self.question_model.generate({"image": question_image, "prompt": prompt})
#         #         print(answer[0])
#         #         prompt = "Qestion: What is the person in the picture doing? Answer:"
#         #         answer =  self.question_model.generate({"image": question_image, "prompt": prompt})
#         #         print(answer[0])
#         return(question_answer)

#     def process_image_text_similarity_itc(self, movie_id, frame):
#         mdfs = {}
#         for mdf in self.get_mdfs(movie_id):
#             if mdf['frame_num'] == frame:
#                 mdfs = mdf
#         mdf_file = self.download_file(mdfs['url']) 
#         # load sample image
#         raw_image = Image.open(mdf_file).convert("RGB")
#         if 1:
#             raw_image.crop()
#             display(raw_image.resize((596, 437)))

#         pass

    def extract_image_from_db_movie_id(self, movie_id, frame):
        mdfs = {}
        for mdf in self.get_mdfs(movie_id):
            if mdf['frame_num'] == frame:
                mdfs = mdf
        return self.process_image_and_captions(mdfs['url'], file_or_url='url')
            
    def process_image_and_captions(self, file, file_or_url='file'):
        # mdfs = {}
        # for mdf in self.get_mdfs(movie_id):
        #     if mdf['frame_num'] == frame:
        #         mdfs = mdf
        if file_or_url == 'url':
            file = self.download_file(file) 
        # load sample image
        raw_image = Image.open(file).convert("RGB")
        #raw_image.crop()
        #display(raw_image.resize((596, 437)))
        question_image = self.question_vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        match_image = self.match_vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        all_captions = "" 
        
        for para in mdfs['paragraphs']:
            #print(para)a
            all_captions = all_captions + " " + para
        all_captions = mdfs['candidate'] + " " + all_captions
        sum_captions = self.sumy_sum(all_captions, 30) 
        #print(sum_captions)
        captions = "" 
        for sentence in sum_captions:
            #for sentence in sentences.split(". "):          
            txt = self.match_text_processors["eval"](sentence)
            itm_output = self.match_model({"image": match_image, "text_input": txt}, match_head="itm")
            itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
            if  itm_scores[:, 1].item() >= 0.3:
                captions = captions +  sentence + "\n"
                #print("Good ", sentence, " ", itm_scores[:, 1].item())
            #else:
                #print("Bad ", sentence, " ", itm_scores[:, 1].item())
        return(question_image, match_image, raw_image, captions, all_captions)

#     def get_questions(self, captions):
#         #Human related question -> ask only if there are people in the image
#         human_related = []
#         human_related.append("Question: How many people are in the picture? Answer: ")
#         human_related.append("Question: How are the people in the picture different from each other? Answer:")
#         human_related.append("Question: What outerwear are the people in the picture wearing? Answer:")
#         #human_related.append("Question: How old is the man shown first from the right in the picture? Answer:")
#         #human_related.append("Question: How old is the man shown second from the right in the picture? Answer:")
#         #human_related.append("Question: What color is the outerwear of the people in the picture? Answer:")
#         #human_related.append("Question: What outerwear do the people in the picture have? Answer:")
#         #human_related.append("Question: What is the person shown first from the right in the picture doing? Answer:")
#         human_related.append("Question: What are the people in the picture doing? Answer:")
#         human_related.append("Question: What are the moods of the people in the picture? Answer:")
#         human_related.append("Question: What are the distinguishing features of people in the picture? Answer:")
#         #Common questions
#         general_prompts = []  
#         general_prompts.append("Question: Where the image is taken in? Answer: ")
#         general_prompts.append("Question: What inanimate objects are in the picture? Answer:")
#         #general_prompts.append("Question: Is there something unusual in the picture? Answer:")
#         general_prompts.append("Question: What is backgound in the picture? Answer:")
#         #Ask GPT-3 for all possible questions, according to captions we have
#         captions = "Captions: \n" + captions + "\n"
#         result  = self.ask_gpt(self.prompt_step1, captions)
#         questions = ''.join([i for i in result if not i.isdigit()])
#         questions = questions.split(".") 
#         for question in questions[1::]:
#             question = question.replace("\n","") 
#             if question:
#                 general_prompts.append("Question: "  + question.replace("\n","") + " Answer:")
#                 #print("Question: " + question + " Answer:")   
#         return(general_prompts, human_related)
    
#     def get_answers(self, generated_questions, human_related, question_image):
#         answers = ""  
#         have_people = "Question: Are there people in the picture? Answer: "
#         answer =  self.question_model.generate({"image": question_image, "prompt": have_people})
#         if answer[0] == "yes":
#             for question in human_related:
#                 prompt = answers + question
#                 answer =  self.question_model.generate({"image": question_image, "prompt": prompt}) 
#                 answers = answers + question + " " + answer[0] + "\n" 
#         for question in generated_questions:
#             prompt = answers + question
#             answer =  self.question_model.generate({"image": question_image, "prompt": prompt})
#             if answer[0] != "no":
#                 answers = answers + question + " " + answer[0] + "\n" 
#         answers = answers + "\n"
#         return(answers)

#     def get_dialog_caption(self, movie_id, frame_num):
#         question_image, match_image, raw_image, captions, all_captions = self.process_image_and_captions(movie_id, frame_num)
#         #print("BEFORE: ", captions)
#         questions, human_related = self.get_questions(captions)
#         answers_all = self.get_answers(questions, human_related, question_image)
#         answers_person = dialog.get_visual_properities(raw_image, movie_id, frame_num)
#         answers = answers_all + answers_person
#         questions_step2 = "Original weak captions: \n" + captions + "\n" + "Clarifying Questions and Answers: \n"+ answers + "\n" + "Accurate captions:"
#         #print(questions_step2)
#         final_results  = self.ask_gpt(self.prompt_step2, questions_step2)
#         return(final_results)