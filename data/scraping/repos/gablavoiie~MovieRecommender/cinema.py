from imdb import Cinemagoer
from coding import *
from coherenlp import *
from vector_score import *
from narrow import *
import pandas

ia = Cinemagoer()


df = pandas.read_csv("movie_data.csv")

def take_input(moviedict):
    moremovies = False

    nlpout = nlp(moviedict["textbox"])
    res = []
    '''
    if moviedict["similar_movies"] != "no":
        moremovies = True
        movies = moviedict["similar_movies"]
        movlist = movies.split(",")
        
        vectors = []
        tags=[]
        for mov in movlist:
            res.append(ia.search_movie(mov)[0]["title"])
            
    '''
    
    if moviedict["genres"] != "no":
        yo = moviedict["genres"]
        if not nlpout[0] in moviedict["genres"]:
            if not nlpout[0] == "no":
                yo = nlpout[0]+","+moviedict["genres"]

        genreforvector = genre_encoding(yo) 
        
    if moviedict["decade"] != "no":
        decadecode = decade_coder(moviedict["decade"].strip())
    else:
        decadecode = 5
       
    moviecode = 1
    if moviedict["length"] != "no":
        length = moviedict["length"].strip().lower()
        if moviedict["length"] == "long" or moviedict["length"] == "longer":
            moviecode = 2
        if moviedict["length"] == "medium":
            moviecode = 1
        if moviedict["length"] == "short" or moviedict["length"] =="shorter":
            moviecode = 0
        else:
            moviecode = 1
    
    vector = [decadecode,moviecode,genreforvector,nlpout[1]]
    
    listofsimilarities = []
    for index in range(1000):
        if narrow(index,moviedict["maturity"],moviedict["actor-director"]):
            listofsimilarities.append((calculate_score(df.loc[index,"Vector"], vector),df.loc[index,"Titles"]))
        
        
        
    listofsimilarities.sort(key = lambda x: x[0],reverse=True)
    
    return(listofsimilarities[0][1])
        
def nlp(text):
    taglist = ['absurd', 'action', 'adult comedy', 'allegory', 'alternate history', 'alternate reality', 'anti war', 'atmospheric', 'autobiographical', 'avant garde', 'blaxploitation', 'bleak', 'boring', 'brainwashing', 'claustrophobic', 'clever', 'comedy', 'comic', 'cruelty', 'cult', 'cute', 'dark', 'depressing', 'dramatic', 'entertaining', 'fantasy', 'feel-good', 'flashback', 'good versus evil', 'gothic', 'haunting', 'historical', 'historical fiction', 'home movie', 'horror', 'humor', 'insanity', 'inspiring', 'intrigue', 'magical realism', 'melodrama', 'murder', 'mystery', 'neo noir', 'non fiction', 'paranormal', 'philosophical', 'plot twist', 'pornographic', 'prank', 'psychedelic', 'psychological', 'queer', 'realism', 'revenge', 'romantic', 'sadist', 'satire', 'sci-fi', 'sentimental', 'storytelling', 'stupid', 'suicidal', 'suspenseful', 'thought-provoking', 'tragedy', 'violence', 'western', 'whimsical']
    g = get_genre_prediction(text)
    t = get_tags(text)
    tag_vector = [0]*len(taglist)
    for index,value in enumerate(taglist):
        if value in t:
            tag_vector[index] = 1
    return(g,tag_vector)
