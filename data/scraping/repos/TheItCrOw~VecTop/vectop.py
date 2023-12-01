import psycopg
import pytextrank
from pgvector.psycopg import register_vector
import spacy
import openai
import numpy as np


class Vectop:

    def __init__(self, openai_api_key, connection_string):
        self.nlp = spacy.load('de_core_news_sm')
        self.nlp.add_pipe("textrank")

        self.max_sent = 5
        openai.api_key = openai_api_key
        self.connection_string = connection_string
        self.ger_eng_channels = {
            # channels
            'Ausland': 'Foreign Countries',
            'Backstage': 'Backstage',
            'Community': 'Community',
            'Familie': 'Family',
            'Fitness': 'Fitness',
            'Geschichte': 'History',
            'Gesundheit': 'Health',
            'International': 'International',
            'Job & Karriere': 'Jobs & Career',
            'Kultur': 'Culture',
            'Mobilität': 'Mobility',
            'Netzwelt': 'Network World',
            'Panorama': 'Panorama',
            'Partnerschaft': 'Partnership',
            'Politik': 'Politics',
            'Psychologie': 'Psychology',
            'Reise': 'Travel',
            'Services': 'Services',
            'Sport': 'Sport',
            'Start': 'Start',
            'Stil': 'Style',
            'Tests': 'Tests',
            'Wirtschaft': 'Economy',
            'Wissenschaft': 'Science',

            # subchannels
            'American Football': 'American Football',
            'Anzeige': 'Advertisement',
            'Apps': 'Apps',
            'Auto-Zubehör': 'Cars',
            'Basketball': 'Basketball',
            'BeyondTomorrow': 'BeyondTomorrow',
            'Bildung': 'Education',
            'Brettspiele': 'Board Games',
            'Business': 'Business',
            'Camping': 'Camping',
            'default': 'default',
            'Deutschland': 'Germany',
            'Diagnose': 'Diagnose',
            'Diagnose & Therapie': 'Diagnose & Therapy',
            'Eishockey': 'Ice Hockey',
            'Elektronik': 'Electronics',
            'Elterncouch': 'Parents',
            'Ernährung & Fitness': 'Nutritions & Fitness',
            'Europa': 'Europe',
            'Europe': 'Europe',
            'Fahrbericht': 'Driving Report',
            'Fahrkultur': 'Driving Culture',
            'Fahrrad & Zubehör': 'Bicycle & Accessories',
            'Fernweh': 'Wanderlust',
            'Formel 1': 'Formula 1',
            'Formel1': 'Formula 1',
            'Fußball-News': 'Soccer News',
            'Gadgets': 'Gadgets',
            'Games': 'Games',
            'Garten': 'Garden',
            'Germany': 'Germany',
            'Gesellschaft': 'Society',
            'Golf': 'Golf',
            'Handball': 'Handball',
            'Haushalt': 'Household',
            'Justiz': 'Law',
            'Justiz & Kriminalität': 'Law & Order',
            'Kino': 'Cinema',
            'Küche': 'Kitchen',
            'Leute': 'People',
            'Ligue 1': 'Ligue 1',
            'Literatur': 'Literature',
            'Medizin': 'Medicine',
            'Mensch': 'Human',
            'Musik': 'Music',
            'Natur': 'Nature',
            'Netzpolitik': 'Network Politics',
            'Olympia': 'Olympics',
            'Premier League': 'Premier League',
            'Primera Division': 'Primera Division',
            'Psychologie': 'Psychology',
            'S-Magazin': 'default',
            'Schwangerschaft & Kind': 'Pregnancy & Children',
            'Serie A': 'Seria A',
            'Sex': 'Sex',
            'Sex & Partnerschaft': 'Sex & Partnership',
            'Soziales': 'Social',
            'Staat & Soziales': 'State & Social',
            'Städte': 'Cities',
            'Städtereisen': 'City Travelling',
            'Technik': 'Technology',
            'Tennis': 'Tennis',
            'Tests': 'Tests',
            'Tomorrow': 'Tomorrow',
            'TV': 'TV',
            'Unternehmen': 'Companies',
            'Unternehmen & Märkte': 'Companies & Markets',
            'Verbraucher & Service': 'Consumers & Service',
            'Web': 'Web',
            'Weltall': 'Space',
            'Wintersport': 'Winter Sports',
            'World': 'World',
            'Zeitgeist': 'Current Mindset',
            'Zeitzeugen': 'Time Witness'
        }
        self.corpus_to_table = {
            'spiegel_sum_1': 'spiegel_embeddings_summarized',
            'times_sum_1': 'times_embeddings'
        }

    def embed(self, text):
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']
        return embeddings

    def tr_summarize(self, text, top):
        ''' summarizes the text with text rank'''
        doc = self.nlp(text)
        summary = ''
        # Create the summary
        for sentence in doc._.textrank.summary(limit_sentences=top):
            summary = summary + str(sentence) + ' '
        return summary.replace('\n', ' ').replace('\r', ' ')

    def get_sim(self, vec, sim_table, take):
        with psycopg.connect(self.connection_string) as conn:
            register_vector(conn)
            sim = conn.execute('SELECT * FROM ' + sim_table + ' ORDER BY embedding <=> %s LIMIT ' + str(take), (vec,)).fetchall() 
            return sim

    def sentinize(self, text):
        '''Checks the content for sentences and summarizes it if needed'''
        sentences = [str(i) for i in self.nlp(text).sents]
        final = ' '.join(sentences)  # This is the text we will extract from
        if(len(sentences) > self.max_sent):
            final = self.tr_summarize(text, self.max_sent)
        return final

    def extract_topics(self, text, language, take, corpus):
        '''Extract topics from a given text'''
        # Language code by: http://www.lingoes.net/en/translator/langcode.htm
        if(language not in ['de-DE', 'en']):
            raise Exception("Currently, only German and English is supported")

        if(corpus not in ['spiegel_sum_1', 'times_sum_1']):
            raise Exception("Unknown corpus: " + str(corpus) + " choose 'spiegel' or 'times'.")

        final = self.sentinize(text)

        # Make the embeddings
        embedded = np.array(self.embed(final.strip()))
        sim = self.get_sim(embedded, self.corpus_to_table[corpus], take)

        # Extract the relevant topics
        tops = {}
        sub_tops = {}
        sources = []
        c = 0
        for vec in sim:
            # Spiegel: Index 4:url, 5: main_topic, index 6: sub_topic, index 3: breadcrumbs
            # Times: Index 3:url, 4: main_topic, index 5: sub_topic, index 2: breadcrumbs
            t = vec[(5 - 1 if corpus == 'times_sum_1' else 5)]
            st = vec[(6 - 1 if corpus == 'times_sum_1' else 6)]
            sources.append(vec[(4 - 1 if corpus == 'times_sum_1' else 4)])

            # Translate the topics for the english language
            if(language == 'en' and corpus != 'times_sum_1'):
                t = self.ger_eng_channels[t]
                st = self.ger_eng_channels[st]

            # We always take the nearest embedding
            if(c == 0):
                tops[t] = 2
                sub_tops[t] = [st]
                c += 1
                continue
            # For the rest we count the occurences and track the subtopics
            if t in tops:
                tops[t] = tops[t] + 1
                li = sub_tops[t]
                li.append(st)
            else:
                tops[t] = 1
                sub_tops[t] = [st]
            c += 1

        # We want only those topics with at least 2 occurences
        final_topics = []
        for k, v in tops.items():
            if(v >= 2):
                final_topics.append([k, list(set([i for i in sub_tops[k] if i != 'default']))])
        return (final_topics, sources)
