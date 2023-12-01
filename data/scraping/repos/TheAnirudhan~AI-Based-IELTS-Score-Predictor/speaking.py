import openai
from pydub import AudioSegment

class Speaking:
    def __init__(self):
        openai.api_key = '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'

    def stt(self,audio_file):
        '''Convertes the First 10 minutes of the audio into text.
        File uploads are currently limited to 25 MB and the following input file types are supported: mp3, mp4, mpeg, mpga, m4a, wav, and webm.'''
        prompt = '''
        This an IELTS speaking test audio of conversation betwwen examiner and test taker. 
        Transcribe exactly word to word
        '''
        # Creating a new file transcript.txt 

        with open('transcript.txt','w') as t:
            t.write('Transcript:\n')
        # Open Audio File
        with open(audio_file, "rb") as audio:      
            transcript = openai.Audio.transcribe("whisper-1", file = audio ,prompt=prompt)
        # print(transcript) #debug
        # saving transcript
        with open('transcript.txt','w') as t:
            t.write(transcript['text']) # type: ignore

    def segment(self,transcript):
        self.features= '''Given a transcript of IELTS speaking test audio of conversation betwwen examiner and test taker.
        you will classify the dialog into examiner and test taker. 
        '''
        self.remarks='''
            {
                 "examiner": {"text" : "<questions>"}
                 "test_taker": {"text" : "<answers>"}
            }
            
           
        '''
        try:
            self.output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.8,
                messages=[
                        {"role": "system",    "content": self.features},
                        {"role": "user",      "content": f"segment the this transcription : {transcript}"},
                        {"role": "assistant", "content": f"{self.remarks}"},
                        
                    ]
            ) 
            print(self.output['choices'][0]['message']['content']) # type: ignore
            self.dialog = eval(self.output['choices'][0]['message']['content']) # type: ignore
            print(self.dialog['test_taker']['text'])
        except Exception as e:
            print(e)
        return self.dialog
            # segements audio as examiner and test taker

    def speaking_predictor(self,dialog): 
        # Predicts the IELTS Score for Coherence, Lexical Resources, Grammatical Range and Accuracy
        self.features= '''Your are an IELTS Speaking grader
                        grade for Coherence, Lexical Resources, Grammatical Range and Accuracy out of 10 with remarks of 100 words
                        Donot give any ranges in the answer. Also give the answer in a 1 point floating decimal
        '''
        
        self.remarks='''{
            "predicted_score" : {"score": "<score>"
                        "remarks": "<overall remarks> "
                        },
            "coherence": {"score":"<score>",
                        "remarks": "<remarks>",
                        "correction": "<correction> "
                        }
            "lexical_resources": {"score":"<score>",
                        "remarks": "<remarks>",
                        "correction": "<correction> "
                        }
            "grammatical_range_and_accuracy": {"score":"<score>",
                        "remarks": "<remarks>",
                        "correction": "<correction> "
                        }
            }
        '''
        try:
            self.output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                # model="text-ada-001",
                # engine="text-ada-001",
                temperature=0.8,
                messages=[
                        {"role": "system",    "content": self.features},
                        {"role": "user",      "content": f"predict score for this transcription : {dialog}"},
                        {"role": "assistant", "content": f"{self.remarks}"},
                        
                    ]
            ) 
            print(self.output['choices'][0]['message']['content']) # type: ignore
            self.out_dict = eval(self.output['choices'][0]['message']['content']) # type: ignore
        except Exception as e:
            print(e)
            self.speaking_predictor(dialog)
        return self.out_dict


#4 band: IELTS speaking test audio of conversation between examiner and test taker Good morning Good morning mam What is your full name? My name is Rohani Sharma What shall I call you? You can call me Rohani Where do you live? I live in Khanna I live in Khanna Ok mam Alright, let's talk about your neighbours Do you know your neighbours? Yes What kind of relationship do you have with your neighbours? Like a brother, sister Try to give answer in full sentence Ok Yes What kind of relationship do you have with your neighbours? My neighbours like These are known as singlings You can say like they are very cooperative They are just like my second family Ok And some of them are just like my siblings Because there are so many elderly people younger than you Yes Ok Ok let's talk about household work Do you usually do any household work? Yes, I like cooking dishes Like so many dishes Is there any household work you dislike? Yes, I dislike to washing clothes I dislike washing clothes, ok Yes Ok Do you like your items being organised or not? Always not organised No, do you like your items organised or not? I like organised but it's not organised anytime It's non-organised household items Ok, my question was like do you like? So you can say yes, I love to organise things But sometimes due to lack of time there is a mess in my room Or at my study place, you can talk about that Yes Ok, alright let's talk about your schedule Are you a morning person? If yes, why? If no, why not? No, I am not a morning person I am very lazy Ok Would you like to change your routine? Yes, I would like to change my routine I change your routine means you want to change my routine I want to change my routine Ok Ok Ruhani Yes Don't be nervous Ruhani, be confident ok Seems you are shivering, so be confident Let's talk about reading, do you love reading books? No, I don't like reading books Why? Because I am not interested in reading books Ok, how often do you buy books? No, I don't buy books You don't buy books, alright Why you don't buy books? As you are a student, so you don't buy any book? No, because I don't like it, like to buy any book Why you don't like? Because it's not my part of my syllabus I can just read only my syllabus books Here I am talking about general books, it's not only about your syllabus or from fiction I was asking you, do you like reading? Being a student of course you read so much But I don't like You can talk about like I don't like other type of books I only read the books related to my curriculum, right? Ok Alright, this is the end of part 1 Now in part 2 I will give you a cue card And you have to prepare for 1 minute You have to speak for 1 to 2 minutes I can stop you in between Ok Is it clear? Yes ma'am Here is your cue card And this is your paper and pencil Ok May I ma'am? I have got many surprises in my life But best surprise is gold ring by my mother And it is beautiful and it given me on my birthday It is costly and I posted pictures on my Instagram And I usually wear it I am very happy to wear it And it is a nice gift to me given to my mother Try to speak more It's all ma'am Try to speak on these parts, ok? Ok ma'am What it was? That you have covered, explain how you felt about that surprise You can add some content in that Ok, it's very good surprise given by my mother And I very like it And it's given me my mother surprisingly But I don't know why she gave me much expensive gift I like it, that's it Never say that's it, ok? And it properly and you have repeated it It given on my birthday It's given me my mother The sentence formation is wrong It was given by your mother on your birthday And you have said I don't know why she gave me much expensive This is wrong So it was a surprise for you And it was a token of love from her You can add on Yes You can add like I was not expecting such an expensive gift from her But I was very happy, I was over the moon after receiving You always wear it, it's a beautiful ring And right now also you are wearing it You can talk about that as well as you are wearing the same ring Fine? And you have to work on your sentence formation There were so many mistakes Even you use the word meaning Yes So make sure you avoid Hindi Punjabi While giving answers in your real exam or in mock test Fine? Can I have paper pencil back? Yes Now in part 3 we will discuss this topic, ok? Yes Do you like surprises? Yeah, I like surprises Avoid your or your friend's exam, ok? Ok ma'am Why do you like surprises? I like surprises because I like to surprise My friends are surprising me With so many gifts I like it Ok You can say I like surprises because Something happens which is not expected Of course it makes you happy, ok? How do people express their happiness? People can not explain me happiness They can feel it Ok What do people do to show that they are happy? They can show your smile And in case you are happy you can express so many Your feelings with attachments Friend Some people are naturally sad and some are happy Why do you think it is? It is nature expression Natural expression it is No, I know Why do you think some people are naturally sad and some are happy? They can... It happens It happens and they can be sad and happy When people are smiling does it always mean they are happy? No, they can be inner happy And they can express so many Your feelings and some people express your feelings But some people don't express your feelings Their feelings not your feelings, ok? Do you think happiness is important? Yes, happiness is important Because they can explain your feeling in happiness Who are they? You are using words like they, your, no You should work on your English Sentence formation as well as on content You need to work a lot, ok? It's fine, you are new in IELTS classes So you are not aware about how the speaking test is conducted I hope this session will help you But let me give you a few tips
# transcript = '''

# Good afternoon. Hello. My name is Jane Smith. What's your name? My name is Faris Ahmadi. You can call me Faris. Can I see your identification please, Faris? Sure. Here you are. Thank you. That's fine. Thank you. Thanks. Now, in part one of the test, I'm going to ask you some questions about yourself. First, do you work or are you a student? I'm a student. What are you studying? I'm studying engineering. Why did you choose to study this subject? I don't really know. My parents thought it would be a good thing for me to study, and I did well at mathematics at high school, so I thought it might be interesting. It's OK, I guess. Have you made many friends on the course? Yes. Yes, I have. There are some great guys on my course. Sometimes we get together and study after class, drink coffee, chat, that kind of thing. Let's talk about keeping fit and healthy now. What do you do to stay healthy? Well, I try to watch my diet. I try to eat healthy foods, you know, fruit, vegetables, but I really love meat too. I think my diet is pretty good. Do you do any exercise? Yeah, sure. From time to time. What kind? Well, I try to go to the gym once or twice a week, and I like going for walks in the evening when it's a bit cooler. How active were you when you were a child? Oh, very active, yes. I was always playing with my brothers, fighting, wrestling, playing games, you know. How important is it for children to be active? Oh, very important. Nowadays kids just play video games all day. They don't get enough exercise, in my opinion. They're just couch potatoes. Now, let's talk about the weather. What kind of weather do you like best? Rainy weather, definitely. Why? Oh, because it rarely rains in my region, so when it does, everyone is really happy, and it cools down the temperature too. Have you noticed any changes in the weather recently in your country? Any changes? Like climate change? Oh, well, I guess it's changed a little. People say it's hotter than it used to be, but I'm not really sure if that's true. What do you like to do in winter? Well, the winter in my country is quite short. It's nice to get outside and go for a walk. You can in winter because it's not so hot. I sometimes go horse riding too. I belong to a club, and I go there a bit at weekends. That's really fun. Thank you. That brings us to the end of part one. Now, Faris, I'm going to give you a topic, and I'd like you to talk about it for one to two minutes. Before you begin, you have one minute to think about what you want to say. You can make some notes if you wish. Do you understand? Yes. OK. You can make some notes on this paper. Here is your topic. I'd like you to talk about a building that you like. Thank you. All right. Now, remember you have one to two minutes for this, so don't worry if I interrupt you. I'll tell you when the time is up. You can begin now. OK. I'm going to tell you about a building I know quite well, and I really like it, actually. Actually, it's in my hometown, Abu Dhabi, and the name of this building is the Capital Gate Building. It's quite a new building, only a few years old, but already it has become an important landmark because it's really unusual in its appearance. The first thing you notice is that it leans over to one side. It's like the building in Italy, the Leaning Tower of Pisa. Only that building is leaning because it's too old. This one is leaning because it was the architect's design. Actually, it's very strange. It's a skyscraper. It's very modern-looking, a really cutting-edge design. It doesn't really have any sharp corners. It has lots of curves, and it is all glass on the outside. OK. Moving on to what it's used for. Well, there is a hotel inside, and maybe some apartments too, and I think there are some offices as well, but I'm not exactly sure. Finally, the reason why I like it is, well, I guess it's because it's unique. It's special. Also, I'm interested in architecture, especially modern architecture, and I'm quite fascinated by how the building stays up. I mean, it doesn't fall down. I think it's because the building has a steel frame, and I heard it also has very deep foundations. So, if you visit my city, I'd recommend that you see this building. That's all, actually. Thank you. Do other people like this building? Yes, everyone likes it. It's a real landmark of the city now. Thank you. Can I have the task card and the paper and pencil, please? Yes, sure. Here you go. Thank you. So, we've been talking about a building you like, and I'd like to discuss with you one or two more general questions related to this. First of all, let's consider traditional and modern buildings. Do you think it's important for the government to preserve traditional buildings, or should the money be spent on essential buildings? Do you think it's important for the government to preserve traditional buildings, or should the money be spent on essential services, such as hospitals and schools? That's a tricky question to answer. Which one is more important? Well, I'd have to say both. First of all, of course, the government should try to maintain traditional buildings. They are our heritage, and they connect us to a past in a way that looking at a photo in a book cannot. Personally, I don't enjoy looking around old buildings such as castles, but I know in some countries tourists flock to these type of buildings, so I can imagine that many people in those countries want to see them preserved. But, on the other hand, we shouldn't be spending so much money on them that we have poor health and education service. Educating our children and treating sick people has to be a priority too. And is it possible for a government to fund all of these things? Can I just check what you mean by fund? Is that pay for? Yes, pay for. Can a government pay for all of these things? I suppose it depends on the country. My country is fairly wealthy, so yes, I think it can afford public services and money for preservation. But in poorer countries I can imagine that the majority of people there would want better schools and good hospitals. Many people like living in modern buildings. Why do you think that is? Well, I think I am one of them. Living in a modern home is much easier on the whole, I think. You don't have to worry about, what's the word, you know, fixing them up all the time.
# '''

if __name__ == '__main__':
    # transcript = open('transcript.txt','r').read()
    # print(transcript)
    obj = Speaking()
    obj.stt("01.mp3")

    # obj.segment(transcript)
    # # obj.dialog = '''My name is Faris Ahmadi. You can call me Faris. Sure. Here you are. Thank you. That's fine. Thank you. Thanks. I'm a student. I'm studying engineering. I don't really know. My parents thought it would be a good thing for me to study, and I did well at mathematics at high school, so I thought it might be interesting. It's OK, I guess. Yes. Yes, I have. There are some great guys on my course. Sometimes we get together and study after class, drink coffee, chat, that kind of thing. Well, I try to watch my diet. I try to eat healthy foods, you know, fruit, vegetables, but I really love meat too. I think my diet is pretty good. Yeah, sure. From time to time. Well, I try to go to the gym once or twice a week, and I like going for walks in the evening when it's a bit cooler. Oh, very active, yes. I was always playing with my brothers, fighting, wrestling, playing games, you know. Oh, very important. Nowadays kids just play video games all day. They don't get enough exercise, in my opinion. They're just couch potatoes. Rainy weather, definitely. Oh, well, I guess it's changed a little. People say it's hotter than it used to be, but I'm not really sure if that's true. Well, the winter in my country is quite short. It's nice to get outside and go for a walk. You can in winter because it's not so hot. I sometimes go horse riding too. I belong to a club, and I go there a bit at weekends. That's really fun. OK. I'm going to tell you about a building I know quite well, and I really like it, actually. Actually, it's in my hometown, Abu Dhabi, and the name of this building is the Capital Gate Building. It's quite a new building, only a few years old, but already it has become an important landmark because it's really unusual in its appearance. The first thing you notice is that it leans over to one side. It's like the building in Italy, the Leaning Tower of Pisa. Only that building is leaning because it's too old. This one is leaning because it was the architect's design. Actually, it's very strange. It's a skyscraper. It's very modern-looking, a really cutting-edge design. It doesn't really have any sharp corners. It has lots of curves, and it is all glass on the outside. Finally, the reason why I like it is, well, I guess it's because it's unique. It's special. Also, I'm interested in architecture, especially modern architecture, and I'm quite fascinated by how the building stays up. I mean, it doesn't fall down. I think it's because the building has a steel frame, and I heard it also has very deep foundations. So, if you visit my city, I'd recommend that you see this building. That's all, actually. Thank you. Yes, both are important. Living in a modern home is much easier on the whole, I think. You don't have to worry about, what's the word, you know, fixing them up all the time.'''
    # obj.speaking_predictor(obj.dialog)

    # print(obj.out_dict)