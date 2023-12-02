import pandas as pd
from .embedding_utils import get_embedding, cosine_similarity
from .toolkit import Tool
from .applenotes import AppleNotes
# from applenotes import AppleNotes
import os
import time
import openai

class NotePad(Tool):  # NotePad now inherits from Tool
    def __init__(self, cache_path='notes_cache.csv'):
        trigger_prompt = None
        actions_prompt = None
        super().__init__(trigger_prompt, actions_prompt)  # Pass trigger_prompt and action_prompt to the Tool's constructor
        self.filepath = cache_path
        self.api = AppleNotes()
        
        if os.path.isfile(self.filepath):
            self.df = pd.read_csv(self.filepath)
            self.df.edit_time = pd.to_datetime(self.df.edit_time)
            self.df.embedding = self.df.embedding.apply(eval)
            self.last_update_time = self.df.edit_time.max()

            self.update(batchsize=2)

        else:
            print(f'file {self.filepath} not found--initializing instead')
            self.initialize()
        
        self.last_update_time = self.df.edit_time.max()
                        
    def handle(self, action, messages, thinker=None):
        
        # add prompt logic to extract description/contents of the note
        
        note_id = None
        WRAPPER = "The AI has retrieved this note:\n\n{}\n\nYou should use the above information inform your responses. Do not directly quote or reference the note, unless the user specifically asks for its contents."
        if action in ['read', 'append']:
            self.update(batchsize=2)
            note = self.search_notes(messages[-1].content).iloc[0]
            thinker.inject(
                WRAPPER.format(note.content),
                meta={'tool':'notepad', 'id':note.id, 'edit_time':note.edit_time}
            )
            note_id = note.id
        
        if action in ['new', 'append']:
            self.write(messages, note_id)
            thinker.inject(
                "Note successfully written to by external process",
                meta={'tool':'notepad'}
            )

        

    def search_notes(self, query, n=1, pprint=False, spectral=False, similarity_fxn=cosine_similarity):
        start_time = time.time()
        query_embedding = get_embedding(
            query,
            engine="text-embedding-ada-002"
        )
        # similarity_fxn = manhattan_distance #cosine_similarity #

        if spectral:
            embs = list(self.df.embedding) + [query_embedding]
            spectral_embedding = SpectralEmbedding(n_components=int(np.sqrt(len(self.df))))
            embs = spectral_embedding.fit_transform(embs)
            self.df['spectral'] = pd.Series(list(embs[:-1]))
            self.df["similarity"] = self.df.spectral.apply(lambda x: similarity_fxn(x, embs[-1]))
        else:
            self.df["similarity"] = self.df.embedding.apply(lambda x: similarity_fxn(x, query_embedding))

        results = (
            self.df.sort_values("similarity", ascending=False)
            .head(n)
        )
        end_time = time.time()
        if pprint:
            print(f'searched {len(self.df)} notes in {end_time-start_time} seconds.\n')

        return results


    def fetch_recent_notes(self, count=5, start=1):
        kwargs = {
            'id_tag':'zmcnuadsl',
            'time_tag':'asdlkjadk',
            'content_tag':'dakjkcmas',
        }
        notedata = self.api.fetch_recent_notes(start=start, count=count, **kwargs) #this api needs to have start index as well
        notelist = self.api.parse_notes(notedata, **kwargs)
        
        df = pd.DataFrame(notelist)
        df.edit_time = pd.to_datetime(df.edit_time)
        
        return df
    
    def initialize(self, n=20):
        notes = self.fetch_recent_notes(n)
        self.df = notes
        self.df['embedding'] = self.df.content.apply(lambda x: get_embedding(x, engine="text-embedding-ada-002"))
        self.last_update_time = self.df.edit_time.max()
        self.df.to_csv(self.filepath, index=False)

    
    def update(self, batchsize=5):
        if not len(self.df):
            print("Call self.initialize() first before calling self.update()")
        start = 1
        notes_pulled = 0
        while True:
            notes = self.fetch_recent_notes(batchsize, start)
            notes = notes[notes.edit_time > self.last_update_time]
            if notes.empty:
                self.df = self.df.sort_values('edit_time', ascending=False).drop_duplicates(subset='id', keep='first').reset_index()[['id', 'edit_time', 'content', 'embedding']]
                self.df.to_csv(self.filepath, index=False)
                return notes_pulled
            notes['embedding'] = notes.content.apply(lambda x: get_embedding(x, engine="text-embedding-ada-002"))
            # self.df = self.df.append(notes, ignore_index=True)
            self.df = pd.concat([self.df, notes], axis=0, ignore_index=True)
            start += batchsize
        
    def write(self, messages, note_id=None):
        prompt = (
            "You are the note-management arm of a superintelligent AI system."
            " Based on the previous conversation, and the user's most recent request,"
            " respond with the content that should be added to the relevant note."
            " Your response will be transcribed directly to the note, so do not"
            " include any explanation of what you are doing, or address the user"
            " in any way. For longer inputs, you should aim to separate ideas"
            " using newlines and indentation."
        )
        
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[message.to_openai() for message in messages[-3:]] + [{'role':'user', 'content':prompt}],
            temperature=0.3,
        )
        content = response['choices'][0]['message']['content']
        
        print(f'writing | {content} | to note {note_id}')
        
        if note_id:
            self.api.append_to_note(note_id, content)
        else:
            self.api.create_new_note(content)
            
    def extend_history(self):
        pass
        
    def rank_notes(self, query):
        pass
    
    #should probably be loaded from a text file or something like that (prompt?)
    def get_trigger_prompt(self):
        prompt = """You are a piece of a superintelligent AI system. Your role is to predict whether the user wants to access their notes app, where they organize their thoughts & manage todo lists.
Example: Tell me an interesting story or a joke.
Prediction: No
Example: How can I create an interactive chart using JavaScript and D3.js?
Prediction: No
Example: Can you find when I last mentioned 'holiday plans' in my saved entries?
Prediction: Yes
Example: What's the weather like today?
Prediction: No
Example: Create a list of potential topics for my upcoming blog posts.
Prediction: Yes
Example: Walk me through setting up a virtual environment for my Python project.
Prediction: No
Example: Write down this idea: 'Start a podcast about technology trends'.
Prediction: Yes
Example: Do I have any information saved about 'product launch strategies'?
Prediction: Yes
Example: Show me the tasks I planned to complete today.
Prediction: Yes
Example: Can you recommend any good books to read?
Prediction: No
Example: Sort my pending tasks based on urgency.
Prediction: Yes
Example: How do I reverse a string in Python?
Prediction: No
Example: Tell me a fun fact about artificial intelligence.
Prediction: No
Example: Could you explain how bubble sort works?
Prediction: No
Example: Jot down my thoughts about the new marketing strategy.
Prediction: Yes
Example: What are the main differences between machine learning and deep learning?
Prediction: No
Example: Add a reminder for the dentist appointment next week.
Prediction: Yes
Example: Share a few tips on how to improve productivity while working from home.
Prediction: No
Example: What were my previous observations on team performance?
Prediction: Yes
Example: What were my key takeaways from last month's meeting?
Prediction: Yes
Example: Could you help me draft an email to my professor, saying that I won't be at this weeks meeting?
Prediction: No
Example: {}
Prediction:"""
        return prompt

    def get_action_prompt(self):
        #I removed all the ones suggesting multiple actions, this could be a cool feature later on
        prompt = """You are a piece of a superintelligent AI system. Your role is to predict whether what action, or combination of actions a user wants to take within their notes app. You can read an existing note, append to an existing note, or create and write to a new note.

Example: Jot down my thoughts about the new marketing strategy.
Prediction: append
Example: Add a reminder for the dentist appointment next week.
Prediction: append
Example: What were my key takeaways from last month's meeting?
Prediction: read
Example: Show me the tasks I planned to complete today.
Prediction: read
Example: Write down this idea: 'Start a podcast about technology trends'.
Prediction: append
Example: Do I have any information saved about 'product launch strategies'?
Prediction: read
Example: What were my previous observations on team performance?
Prediction: read
Example: Create a list of potential topics for my upcoming blog posts.
Prediction: new
Example: Sort my pending tasks based on urgency.
Prediction: append
Example: Can you find when I last mentioned 'holiday plans' in my saved entries?
Prediction: read
Example: Can you create a new note with my thoughts about the project meeting today?
Prediction: new
Example: Add 'buy groceries' to my todo list.
Prediction: append
Example: Please show me the notes from yesterday's brainstorming session.
Prediction: read
Example: What's on my task list for this week?
Prediction: read
Example: Remind me to call the doctor tomorrow by adding a note.
Prediction: new
Example: Search my notes for any mention of 'vacation plans'.
Prediction: read
Example: Read me the last note I created.
Prediction: read
Example: Add a new section in my project ideas note with the title 'Mobile App Design'.
Prediction: append
Example: {}
Prediction:"""
        return prompt
        
            
