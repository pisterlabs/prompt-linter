import os
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,  LLMChain

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np
import sys




def initialize_model(template, input_variables):

  # llm = ChatOpenAI(model_name="gpt-3.5-turbo")
  llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")

  prompt = PromptTemplate(template= template, input_variables= input_variables)

  llm_chain = LLMChain(prompt=prompt, llm=llm)

  return llm_chain

#@title QA_Pipeline_TfIdf { form-width: "20%" }





class QA_Pipeline_TfIdf:
    ## call "get_nearest_songs" to get the nearest songs and it will intialie tfidf and call all the rest of the functions internally.
    # still, if you want to call the functions individually then call them in the following order:
    # 1. createTfIdfTable
    # 2. RunTfIdf
    


    def __init__(self):
        pass        

    def normalizeData(self,data,isNormalizeAgainstMax=False):
        if sum(data)==0:
            return data
        if isNormalizeAgainstMax:
          return [float(i)/max(data) for i in data]  # to normalize against max
        else:
          return [float(i)/sum(data) for i in data]  # to normalize to make sum = 1
        

    def create_tfidf_features(self,data, max_features=5000, max_df=0.85, min_df=2):
        """ Creates a tf-idf matrix for the `data` using sklearn. """
        tfidf_vectorizor = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word',
                                        stop_words='english', ngram_range=(1, 1), max_features=max_features,
                                        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                                        max_df=max_df, min_df=min_df)
        X = tfidf_vectorizor.fit_transform(data)
        print('tfidf matrix successfully created.')
        return X, tfidf_vectorizor

    def calculate_similarity( self,query, top_k=20):
        """ Vectorizes the `query` via `vectorizer` and calculates the cosine similarity of
        the `query` and `allDocuments` (all the documents) and returns the `top_k` similar documents."""
        
        # Vectorize the query to the same length as documents
        query_vec = self.vectorizer.transform(query)
        # Compute the cosine similarity between query_vec and all the documents
        cosine_similarities = cosine_similarity(self.tfidfTransformed_docs,query_vec).flatten()

        # Sort the similar documents from the most similar to less similar and return the indices
        most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]

        # Sort the similar documents from the most similar to less similar and return the scores
        cosine_similarities = np.sort(cosine_similarities)[:-top_k-1:-1]

        # #normalize scores
        cosine_similarities = self.normalizeData(cosine_similarities,isNormalizeAgainstMax=True)
        return most_similar_doc_indices, cosine_similarities

    def RunTfIdf(self,question,top_n=20, getDocuments=False,data=None):
        '''
        given a question find the top_n most similar documents.
        if document text is also needed then pass getDocuments=True and pass the data which was originally passed to createTfIdfTable
        '''
        top_idx,cosine_similarities = self.calculate_similarity( [question],top_k=top_n)
        
        if getDocuments:
            if data==None:
                sys.exit('data to be returned but no data provided. data is n')
            retData = [data[i] for i in top_idx]
            return top_idx,cosine_similarities,retData 
        else:
            return top_idx,cosine_similarities

    def createTfIdfTable(self,data,maxFeatures=10000):
        self.tfidfTransformed_docs,self.vectorizer = self.create_tfidf_features( data  ,max_features=maxFeatures)
        # features = vectorizer.get_feature_names()

    def get_nearest_songs(self, data_df, text, top_n=5):
        '''
        given a string text, fine the nearest songs descriptions from the data
        return a list of descriptions and lyrics.

        @params:
            data_df: pandas dataframe: dataframe containing the data
            text: string : the query
        '''

        desc_data  = data_df['description'].tolist()


        self.createTfIdfTable(desc_data,maxFeatures=10000)

        top_idx,cosine_similarities,retData  = self.RunTfIdf(text, top_n=top_n, getDocuments=True, data=desc_data)

        # get the descriptions of top_idx from df
        desc = []
        lyrics = []
        titles = []
        ids = []

        df_titles = data_df['title'].tolist()
        df_ids = data_df['id'].tolist()
        df_desc = data_df['description'].tolist()
        df_lyrics = data_df['lyrics_clean_with_newline'].tolist()


        for i in top_idx:
            desc.append(df_desc[i])
            lyrics.append(df_lyrics[i])
            titles.append(df_titles[i])
            ids.append(df_ids[i])

        return ids, titles, desc, lyrics
    

def generate_lyrics_internal(desc_user, data_df, specific_artist=None ):
    '''
        Given the user query/descipriton of the lyrics, return the lyrics of the song
        This is the function that is called when user asks to generate the lyrics
        @params:
            desc_user : string : the description of the song that the user wants to generate. this is the user input
            data_df : pandas dataframe : the dataframe containing the data
            specific_artist : string : the name of the artist. if None then take the whole data. if not None then take only the data of this artist
        @returns:
            model_output : string : the output of the model
    '''

        
    # set the openai api key - below one is changed for security reasons
    os.environ['OPENAI_API_KEY']= "asdqweqwe"

    template="""You are a English song Lyricist. Famous singers come to you with descriptions of the kind of song they want you to write.
    They also give you some examples of the song lyrics based on the description. 
    Output only the lyrics of the song that you should write. do not output anything else.
    The avg number of words in lyrics should be around 500-600 and it should have some chorus and verses.

    Format of the examples is:
    -- Description: <description of the song>
    -- Lyrics : <lyrics of the song>

    ----------------------------------- EXAMPLES START -----------------------------------
    -- Description 1: ```{desc_1}``` 
    -- Lyrics 1: ```{lyrics_1}``` 

    -- Description 2: ```{desc_2}``` 
    -- Lyrics 2: ```{lyrics_2}``` 

    -- Description 3: ```{desc_3}``` 
    -- Lyrics 3: ```{lyrics_3}``` 

    -- Description 4: ```{desc_4}``` 
    -- Lyrics 4: ```{lyrics_4}``` 

    -- Description 5: ```{desc_5}``` 
    -- Lyrics 5: ```{lyrics_5}``` 

    ----------------------------------- EXAMPLES END -----------------------------------

    Note that the avg number of words in lyrics should be around 500-600 and it should have some repeating chorus and verses. 

    -- Description: ```{desc_user}``` 

    -- Lyrics : 

    """

    input_variables = ["desc_1", "lyrics_1", "desc_2", "lyrics_2", "desc_3", "lyrics_3", "desc_4", "lyrics_4", "desc_5", "lyrics_5", "desc_user"] # parameters. for tempalte


    # if specific_artist is None or empty
    if specific_artist is None or specific_artist == "":
        sub_df = data_df

    else:
        # conver the specific_artist into Proper Case
        # specific_artist is a string
        specific_artist = specific_artist.title()

        # repalce "Format of the examples is:" with "the Lyrics should be in the style of <artist_name>." + "\n Format of the examples is:"
        temp = "Format of the examples is:"
        template = template.replace(temp, "the Lyrics should be in the style of " + specific_artist + ".\n" + temp)

        try:
            sub_df = data_df[data_df['artist']==specific_artist]
            # if this artist has less then 5 entries then get the whole df
            if sub_df.shape[0] < 5:
                print("This artist has less than 5 songs. taking the whole data")
                sub_df = data_df 
        except:
            print("Artist wasnt found. taking the whole data")
            sub_df = data_df
    

    llm_chain = initialize_model(template, input_variables)


    tfidf = QA_Pipeline_TfIdf()

    ids, titles, desc, lyrics = tfidf.get_nearest_songs(sub_df, desc_user)

    params = {}
    params["desc_1"] = desc[0]
    params["desc_2"] = desc[1]
    params["desc_3"] = desc[2]
    params["desc_4"] = desc[3]
    params["desc_5"] = desc[4]
    params["lyrics_1"] = lyrics[0]
    params["lyrics_2"] = lyrics[1]
    params["lyrics_3"] = lyrics[2]
    params["lyrics_4"] = lyrics[3]
    params["lyrics_5"] = lyrics[4]

    params["desc_user"] = desc_user

    
    model_output = llm_chain.run(params)

    return model_output