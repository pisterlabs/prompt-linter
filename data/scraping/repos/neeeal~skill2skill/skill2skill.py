from gensim.models import KeyedVectors
import cohere
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def load_wv(path="wordvectors\word2vec.wordvectors"):
    ## Load word vectors
    wv = KeyedVectors.load(path, mmap='r')
    return wv

def preprocess_input(array, wv):
    ## get only words in vocabulary of models
    return [word for word in array if word in wv.key_to_index]

def get_most_similary(input_array, wv, topn=10):
    print("Getting similarity..")
    ## get n most similar words to input_array
    sims = wv.most_similar(input_array, topn=topn)
    output = {'input':input_array,'similars':{sim[0]:sim[1] for sim in sims}}
    return output
    
def get_skills_from_desc(desc, stop_words= stopwords.words('english')):
    print("Extracting skills from descriptions...")
    ## Use Cohere LLM to generate skills with description as prompt
    co = cohere.Client('UVX7wrNGRe6OZ2q8FvNEg90YhKuweHTjG6CgvZyt')
    response = co.generate(
        prompt="If I'm finding a collaborator for my project, list down top 32 relevant skills I should look out for. Don't add explanations. Below is the description of my project:"+desc,
    )
    ## Clean up skills
    skills=[]
    for n,desc in enumerate(response[0].split('. ')):
        print(n)
        temp = re.sub(r"[^ a-zA-Z]+",'', desc).lower()
        temp = [word for word in temp if word not in stop_words if len(word) > 1]
        if len(temp) > 0: skills.append(temp)
    return skills

if __name__ == '__main__': 
    import re

    # def input_to_array(input, stop_words = stopwords.words('english')):
    #     ## Retrieving job descriptions
    #     ## Applying NLP: lowercase, remove non-alphanum, remove stopwords
    #     temp = re.sub(r"[^ a-zA-Z]+",' ', input).lower().split(' ')
    #     temp = [word for word in temp if word not in stop_words if len(word) >= 1]
    #     return temp

    wv = load_wv()

    while True:
        input_array = get_skills_from_desc('minimum years experience sdet role mobile test automation ios android years experience selenium cucumber front end automation proficient mobile operating systems apple ios android perform tests experience testing frameworks xcuitest appium education bachelor degree computer science technical field equivalent work thank prasad job title sdet software developer engineer test senior mobile development engineer test employment type months location austin westlake candidates job description experience xcui espresso hands coding experience java kotlin swift object oriented programming languages working knowledge industry standard tools logging bugs managing test cases jira mtm zephyr hpq proven ability successfully balance deliver mul job sc desktop support specialist location woodlands tx type contract client seeking desktop support professional join team great opportunity professionals passion providing customer service excellence also want grow career looking new opportunity want work excellent organization send us resume key tasks provide effective support client environment including rest position objective provides technical assistance support incoming queries issues related computer systems software hardware responsible providing technical assistance support related computer systems hardware software responds queries runs diagnostic programs isolates problem determines implements solution maintains daily performance end user computers peripherals walk clients problem solving processes related use com title systems engineering plant network engineer description stg fast growing digital transformation services company providing fortune companies digital transformation mobility analytics cloud integration services information technology engineering product lines stg repeat business rate existing clients achieved industry awards recognition services crain detroit business named stg michigan fastest growing description responsibilities managing software release process driver team working infrastructure teams ensure automated systems run continuously coordinating qa teams ensure adequate test coverage automation systems managing optimizing test sets used various points integration monitoring optimizing time taken execute test sets enabling additional automated test capabilities beyond current featureset clear commun role tester location remote duration months contract education bachelors description tester ability validate front end features compliance web accessibility standards tester experience testing front end applications using accessibility tools jaws zoomtext dragon candidate also experience testing accessibility mobile devices ios android using ios voiceover android talkback working knowle successful candidate responsible providing timely computer related support diverse set users maintaining high level customer satisfaction ideal candidate must strong written verbal communications skills responsibilities end user support diagnosis user support problems via walk ins telephone remote support users experiencing problems work site requires extensive knowledge computer system questions whi job sc desktop support location woodlands tx type contract reporting houston infrastructure end user services manager candidate responsible providing consistent high quality service ensuring positive client experience using services deskside support representative dsr responsible communicating assisting clients related issues requests including hardware software mobile devices remote access meeting con tech lead full stack web developers help create next generation online internet banking application framework uiyou design develop new features world class internet banking platform written net developers work frameworks design sdks well components written net net corewe looking creative skilled people passionate development opportunity help design share ideas new approaches software development engineer full stack web developers help create next generation online internet banking application framework uiyou design develop new features world class internet banking platform written net developers work frameworks design sdks well components written net net corewe looking creative skilled people passionate development opportunity help design share ideas position information position title desktop support technician position summary desktop technician site admin would essentially entail installing equipment troubleshooting hardware software issues working support areas troubleshooting user account issues local site required remote across divers set clients user base essential duties responsibilities advising staff appropriate procedures directing related queries recommen job title wireless test engineer location san jose ca duration months contract job description samsung soc lab san diego looking cellular engineer lab field test join team passionate engineers start spirit debug test samsung world class cellular modem solution candidate expected bring innovation strong execution skills cellular group lte lte fr fr testing lab carrier iot field certification triveni technology company located new york city area triveni utilizes agile technologies develop solutions clients seeking highly motivated engineers join team successful candidate focus engineering development complex business requirements provide casual work environment hard work rewarded job title sr java tech lead location pleasanton california duration months plus experience years visa')
        if input_array[0] == 'q': break
        processed_input = preprocess_input(input_array, wv)
        sims = get_most_similary(processed_input, wv)
        print(sims)

