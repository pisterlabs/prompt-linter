
import re
import os

from anthropic import HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv

load_dotenv()

LEARNING_LANGUAGE = os.getenv("LEARNING_LANGUAGE")
NATIVE_LANGUAGE = os.getenv("NATIVE_LANGUAGE")

def extract_text(tag, string):
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, string, re.DOTALL)
    return match.group(1).strip() if match else None

def create_conversation_prompt(message, chat_id):
        # # Initialize history list if not present for the chat_id
        # if chat_id not in chat_histories:
        #     chat_histories[chat_id] = []
        # Add the new message to the history
        # chat_histories[chat_id].append(message)
        # # Format the history for the prompt
        # formatted_history = "\n".join(
        #     f"{HUMAN_PROMPT} {msg}" for msg in chat_histories[chat_id])

        prompt_components = {
            "human_prompt": HUMAN_PROMPT,
            "role": f"You will be acting as a highly skilled and helpful {LEARNING_LANGUAGE} teacher. Your goal is to engage in conversation with a beginner student to help them learn and practice new vocabulary. The student is practicing everyday conversation and you are pretending to be their friend.",
            "tone": "You will use a friendly tone.",
            "rules": f"""
                Here are some important rules for the interaction:
                - You will always use simple language.
                - You will respond to the student with a correction containing a glossary for words they didn't know and a correction of their phrase.
                - If the student uses {NATIVE_LANGUAGE} words, you will add a glossary to your correction with the word and its translation.
                - In your response, you will provide a glossary introducing the key word and it's translation.
                - Your response will not be longer than 3 sentences.
                """,
            "example": """
                Here is an example interaction for a french teacher speaking to a student who's native language is english:
                <example>
                    Human: Je suis Amber. J'aime read des livres et travel.
                    Assistant: 
                    <correction> 
                        <glossary>
                            travel: voyager
                            read:lire
                        </glossary>
                        <corrected_response>
                            J'aime <b>lire</b> des livres et voyager.
                        </corrected_response>
                    </correction>
                    <response> 
                        <glossary> 
                            intéressant: interesting
                            où: where 
                        </glossary> 
                            Bonjour Amber. Enchantée ! C'est très intéressant que tu aimes lire et voyager. Où est-ce que tu aimes voyager ?
                        </response>
                </example>
                """,
            "input": f"Human: {message}",
            "instructions": f"""
                Fill in the following tag structure with your response:
                    <correction>
                        <correction_glossary>
                        </correction_glossary> 

                        <corrected_response>
                        </corrected_response>
                    </correction>

                    <bot_response>
                        <response_glossary>
                        </response_glossary>

                        <response>
                        </response>
                    </bot_response>

                Think step by step:
                    - Extract {NATIVE_LANGUAGE} words and build the correction_glossary with {LEARNING_LANGUAGE} word: {NATIVE_LANGUAGE} translation
                    - Write the corrected_response with the words from the glossary in <b>bold</b>
                    - Write the response
                    - Build the response_glossary with key vocabulary (less than 3 words)
                    - Put the key vocabulary in the response in bold
                """,
            "AI prompt":f"{AI_PROMPT} <correction> <correction_glossary>"
        }
        prompt = '\n'.join(prompt_components.values())
        return prompt
    
def format_conversation_response(raw_response):
    raw_response = "<correction> <correction_glossary>" + raw_response
    tags = ["correction_glossary", "corrected_response", "response_glossary", "response"]
    [correction_glossary, corrected_response, response_glossary, response] = [extract_text(tag, raw_response) for tag in tags]
    
    # # Add Claude's response to the history as well
    # chat_histories[chat_id].append(raw_response)

    formatted_response = f"<b> {correction_glossary} </b>\n\n{corrected_response}\n\n\n{response_glossary}\n\n {response}"
    return formatted_response

def create_news_prompt(message, chat_id):
    prompt_components= {
          "human_prompt": HUMAN_PROMPT,
          "role": "You will be acting as a highly skilled French teacher. Your goal is to summarise news articles at the student's language level (A1) to teach them new vocabulary and give them reading practice.\n",
          "rules":"""Here are some important rules for the interaction:
            - Your summaries should never be more than 4 sentences long and each sentence should be on a new line.
            - Your sentences should be as short as possible.
            - Your summaries should always introduce between 2 and 6 words
            - You will provide a vocab glossary before the summary with new words and their translation""",
          "example":"""Here is an example of how to do a glossary and summary from an article:
            <example>
                <article> 
                Volodymyr Zelensky conteste que la guerre en Ukraine se trouve dans une \"impasse\"\nDans le cadre d'une visite de la présidente de la Commission européenne samedi à Kiev, le président ukrainien a contesté que le conflit de son pays avec la Russie se trouve dans une \"impasse\", et déploré que la guerre au Proche-Orient détourne l'attention du monde.\n\nVolodymyr Zelensky accueille Ursula von der Leyen, la présidente de la Commission européenne, à Kiev, pour la sixième fois depuis le début du conflit avec la Russie, le 4 novembre 2023. © Présidence ukrainienne via AFP\n\nMalgré les déclarations pessimistes d'un haut gradé ukrainien, le président Volodymyr Zelensky n'en démord pas : la guerre contre la Russie n'est pas dans une \"impasse\". C'est ce qu'il a déclaré samedi 4 novembre au cours d'une conférence de presse à Kiev avec la présidente de la Commission européenne Ursula von der Leyen, venue discuter du chemin d'adhésion de l'Ukraine à l'UE.\n\n\"Le temps a passé aujourd'hui et les gens sont fatigués (...). Mais nous ne sommes pas dans une impasse\", a affirmé Volodymyr Zelensky, alors qu'un commandant de haut rang a affirmé cette semaine que les deux armées se trouvaient prises au piège d'une guerre d'usure et de positions.\n\nL'Ukraine mène depuis juin une lente contre-offensive pour tenter de libérer les territoires occupés de l'Est et du Sud. Mais, jusqu'ici, les avancées ont été très limitées. La ligne de front, longue de plus de 1 000 km, n'a guère bougé depuis près d'un an et la libération de la ville de Kherson en novembre 2022.\n\nÀ lire aussi**\"Impasse\" tactique, pessimisme occidental... Kiev en quête d'un second souffle face à la Russie**\n\nLe Kremlin avait aussi assuré jeudi que le conflit en Ukraine ne se trouvait pas dans une \"impasse\", contestant des propos du commandant en chef de l'armée ukrainienne, Valery Zaloujny, dans un entretien à The Economist.\n\n\"Tout comme lors de la Première Guerre mondiale, nous avons atteint un niveau technologique tel que nous nous trouvons dans une impasse\", avait déclaré Valery Zaloujny à l'hebdomadaire britannique. \"Il n'y aura probablement pas de percée magnifique et profonde\", avait-il ajouté.\n\n\"Nous allons relever ce défi\"\nLe président Volodymyr Zelensky a démenti toute pression des pays occidentaux pour entamer des négociations avec la Russie. Il a admis que le conflit entre Israël et le mouvement islamiste palestinien Hamas avait \"détourné l'attention\" de la guerre opposant l'Ukraine à la Russie.\n\n\"Nous nous sommes déjà retrouvés dans des situations très difficiles lorsqu'il n'y avait presque aucune focalisation sur l'Ukraine\", a noté le président ukrainien, ajoutant : \"Je suis absolument certain que nous allons relever ce défi.\"\n\nLes soutiens de l'Ukraine, en particulier les États-Unis, répètent qu'ils procureront de l'aide militaire et financière à Kiev jusqu'à la défaite de la Russie.\n\nLors de cette sixième visite d'Ursula von der Leyen en Ukraine depuis le début de l'invasion russe en février 2022, la dirigeante compte aborder le \"soutien militaire\" des Européens, ainsi que \"le douzième paquet de sanctions\" de l'UE contre la Russie, en cours de préparation, a-t-elle déclaré à des journalistes.\"
                </article>
                
                <glossary>
                    impasse: dead end
                    etre d'accord: agree with
                    un conflit: a conflict
                    une defaite: a defeat
                    plutot que: rather than
                </glossary>
                <summary> 
                Un general ukrainien a dit que la guerre est a une impasse. \nZelensky n'est pas d'accord.\nZelensky est triste que les gens regardent le conflit entre Israël et la Palestine plutôt que la guerre en Ukraine.\nLes soutiens de l'Ukraine en Occident disent qu'ils aideront Kiev jusqu'à la défaite de la Russie.
                </summary>
                </example>
                
                <example>
                <article> 
                    Au Népal, un séisme de magnitude 5,6 fait plus d'une centaine de morts\nSelon un nouveau bilan provisoire communiqué samedi par les autorités népalaises, au moins 132 personnes sont mortes et plus de 100 ont été blessées dans un tremblement de terre qui a secoué une région reculée du Népal, où les secours s'organisent à la recherche des survivants.\n\nPublié le : 04/11/2023 - 07:30\n\n3 mn\nCette photo fournie par le bureau du Premier ministre népalais montre une zone touchée par le tremblement de terre dans le nord-ouest du Népal, le 4 novembre 2023.\nCette photo fournie par le bureau du Premier ministre népalais montre une zone touchée par le tremblement de terre dans le nord-ouest du Népal, le 4 novembre 2023. © AP\nPar :\nFRANCE 24\nSuivre\nSéisme meurtrier au Népal. Au moins 132 personnes sont mortes dans un tremblement de terre qui a secoué une région reculée du pays, selon un nouveau bilan provisoire communiqué samedi 4 novembre par les autorités népalaises. Sur place, les secours s'organisent à la recherche des survivants.\n\nLe séisme de magnitude 5,6 a été mesuré à une profondeur de 18 km selon l'Institut américain d'études géologiques USGS. Il a frappé l'extrême ouest du pays himalayen tard vendredi soir. Son épicentre a été localisé à 42 km au sud de Jumla, non loin de la frontière avec le Tibet.\n\n\"92 personnes sont mortes à Jajarkot et 40 à Rukum\", a déclaré à l'AFP le porte-parole du ministère de l'Intérieur, Narayan Prasad Bhattarai, citant les deux districts à ce stade les plus touchés par le séisme, situés au sud de l'épicentre dans la province frontalière de Karnali.\n\nPlus de 100 blessés ont été dénombrés dans ces deux districts, a pour sa part affirmé le porte-parole de la police népalaise, Kuber Kathayat.\n\n\"Certaines routes sont bloquées à cause des dégâts\"\nLes forces de sécurité népalaises ont été largement déployées dans les zones touchées par le séisme pour participer aux opérations de secours, selon le porte-parole de la police de la province de Karnali, Gopal Chandra Bhattarai.\n\n\"L'isolement des districts rend difficile la transmission des informations\", a-t-il ajouté. \"Certaines routes sont bloquées à cause des dégâts, mais nous essayons d'atteindre la zone par d'autres voies.\"\n\nÀ Jajarkot, l'hôpital de secteur a été pris d'assaut par les habitants y transportant des blessés.\n\nDes vidéos et des photos publiées sur les réseaux sociaux montrent des habitants fouillant les décombres dans l'obscurité pour extraire des survivants des constructions effondrées. On y voit des maisons en terre détruites ou endommagées et des survivants à l'extérieur pour se protéger de possibles autres effondrements, tandis qu'hurlent les sirènes des véhicules d'urgence.\n\nLe Premier ministre népalais, Pushpa Kamal Dahal, est arrivé samedi dans la zone touchée après avoir exprimé \"sa profonde tristesse pour les dommages humains et physiques causés par le tremblement de terre\".\n\nLe Népal sur une faille géologique majeure\nDes secousses modérées ont été ressenties jusqu'à New Delhi, la capitale de l'Inde située à près de 500 km de l'épicentre.\n\nLe Premier ministre indien Narendra Modi s'est dit \"profondément attristé\" par les pertes humaines au Népal. \"L'Inde est solidaire du peuple népalais et est prête à lui apporter toute l'aide possible\", a-t-il ajouté.\n\nDeeply saddened by loss of lives and damage due to the earthquake in Nepal. India stands in solidarity with the people of Nepal and is ready to extend all possible assistance. Our thoughts are with the bereaved families and we wish the injured a quick recovery. @cmprachanda\n\n— Narendra Modi (@narendramodi) November 4, 2023\nLe résumé de la semaine\nFrance 24 vous propose de revenir sur les actualités qui ont marqué la semaine\n\nJe m'abonne\nLes séismes sont fréquents au Népal, qui se trouve sur une faille géologique majeure où la plaque tectonique indienne s'enfonce dans la plaque eurasienne, formant la chaîne de l'Himalaya. La secousse a été suivie plusieurs heures après par des répliques de magnitude 4 dans le même secteur, selon l'USGS.\n\nPrès de 9 000 personnes sont mortes en 2015 lorsqu'un tremblement de terre de magnitude 7,8 a frappé le Népal, détruisant plus d'un demi-million d'habitations et 8 000 écoles.\n\nDes centaines de monuments et de palais royaux – dont des sites de la vallée de Katmandou, classée au patrimoine mondial de l'Unesco et attirant des touristes de toute la planète – avaient subi des dégâts irréversibles, donnant un grand coup au tourisme népalais.\n\nEn novembre 2022, un séisme de magnitude 5,6 avait fait six morts dans le district de Doti, près du district de Jajarkot frappé vendredi soir.
                </article>
                <glossary>
                    tremblement de terre: earthquake
                    un blessé: an injured person
                    la zone touchée: affected area
                    dégâts: damage
                </glossary> 
                <summary>
                    Il y a eu un tremblement de terre au Népal.
                    Plus de 100 personnes sont mortes.
                    Plus de 100 personnes sont blessées. 
                    Les routes sont bloquées dans la zone touchée. 
                    Il y a beaucoup de dégâts.
                </summary>
                </example>""",
            "instructions": """Here is the article to summarise for your student:
                <article>
                    Rugby : les Bleues s'inclinent face au Canada pour leur dernier match du WXV et terminent à la 5e place\nAprès leur défaite face à l'Australie la semaine passée, les Bleues se sont de nouveau inclinées samedi face au Canada (20-29), pour leur dernier match du WXV.\n\nArticle rédigé parfranceinfo: sport\nFrance Télévisions - Rédaction Sport\nPublié le 04/11/2023 08:53\nMis à jour le 04/11/2023 09:15\nTemps de lecture : 1 min\nLes Françaises ont été dominées par les Canadiennes, dans leur dernier match du WXV, à Auckland le 4 novembre 2023. (WORLD RUGBY)\nLes Françaises ont été dominées par les Canadiennes, dans leur dernier match du WXV, à Auckland le 4 novembre 2023. (WORLD RUGBY)\nNouveau coup dur pour les Bleues du XV de France. Après s'être inclinées face à l'Australie la semaine dernière, les Françaises ont encaissé un second revers de suite face au Canada, sur le même score (20-29), samedi 4 novembre à Auckland (Nouvelle-Zélande), pour leur dernier match du WXV, nouveau tournoi international. Une défaite décevante alors que les Bleues avaient largement battu les Canadiennes (36-0) lors du match pour la troisième place du Mondial il y a presque un an.\n\n\nUne 5e place décevante\nPourtant bien entrées dans la partie avec un essai de Pauline Bourdon Sansus (14e) et menant à la pause (10-7), les Bleues sont retombées dans leurs travers, faisant preuve d'indiscipline et d'irrégularité. Les Canadiennes, elles, n'ont pas vacillé et ont enchaîné les essais avec Emily Tuttosi (35e) et Krissy Scurfield (43e et 51e).\n\nMalgré un deuxième essai tricolore, inscrit par Marine Ménager (57e), les Françaises ont commis trop de fautes (12 pénalités concédées) et d'approximations pour espérer la victoire. Après la défaite contre l'Australie, la France s'incline ainsi une nouvelle fois dans la compétition. Une défaite inattendue après le succès historique deux semaines plus tôt face à la Nouvelle-Zélande (18-17), championne du monde en titre. Le XV de France termine ainsi à une décevante 5e place la première édition du WXV. L'Angleterre a remporté la compétition avec trois victoires en autant de matches. 
                </article>
                Put your response in <response></response> tags.""",
            "AI prompt": "{AI_PROMPT} <response>"
     }
    prompt = '\n'.join(prompt_components.values())
    return prompt

def format_news_response(raw_news): 
    tags = ["glossary", "summary"]
    [glossary, response] = [extract_text(tag, raw_news) for tag in tags]
    return f"<u><b>News Article</b></u> /n/n {glossary} /n/n {response}"
     
     
