import logging
import os
import re

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Milvus

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the logging format
)


class QueryHandler:
    def __init__(self, openai_api_key, milvus_api_key):
        load_dotenv()  # take environment variables from .env.
        self.openai_api_key = openai_api_key
        self.milvus_api_key = milvus_api_key

        connection_args = {
            "uri": "https://in03-5052868020ac71b.api.gcp-us-west1.zillizcloud.com",
            "user": "vaclav@pechtor.ch",
            "token": self.milvus_api_key,
            "secure": True,
        }

        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.milvus = Milvus(
            embedding_function=self.embeddings,
            collection_name="LuGPT",
            connection_args=connection_args,
        )
        self.chat_history = []


    def process_output(self, output):
        # Check if 'SOURCES: \n' is in the output
        if 'QUELLEN:' in output['answer']:
            # Split the answer into the main text and the sources
            answer, raw_sources = output['answer'].split('QUELLEN:', 1)

            # Split the raw sources into a list of sources, and remove any leading or trailing whitespaces
            raw_sources_list = [source.strip() for source in raw_sources.split('- ') if source.strip()]

            # Process each source to turn it back into a valid URL
            sources = []
            for raw_source in raw_sources_list:
                if raw_source:  # Ignore empty strings
                    # Remove the prefix and the ending '.txt' and replace '__' with '/'
                    valid_url = 'https://' + raw_source.split('/')[-1].replace('__', '/').rstrip('.txt\n')
                    sources.append(valid_url)
        else:
            # If there are no sources, return the answer as is and an empty list for sources
            answer = output['answer']
            sources = []

        # Join the sources list into a single string with each source separated by a whitespace
        sources = ' '.join(sources)

        return answer, sources


    def get_answer(self, query, history):

        qa_prompt = """Angesichts der folgenden extrahierten Teile eines langen Dokuments und einer Frage, erstelle eine abschließende Antwort mit Verweisen ("SOURCES").
        Wenn Du die Antwort nicht kennst, sag einfach, dass Du es nicht weißt. Versuche nicht, eine Antwort zu erfinden.
        Gib IMMER einen "SOURCES"-Teil in Deiner Antwort zurück.

        FRAGE: Welches Landes-/Staatsrecht regelt die Auslegung des Vertrages?
        =========
        Content: Dieser Vertrag wird durch englisches Recht geregelt und die Parteien unterwerfen sich der ausschließlichen Gerichtsbarkeit der englischen Gerichte in Bezug auf jeden Streit (vertraglich oder nicht vertraglich) in Bezug auf diesen Vertrag, es sei denn, eine Partei kann sich an jedes Gericht wenden, um eine einstweilige Verfügung oder andere Rechte zum Schutz ihrer geistigen Eigentumsrechte zu beantragen.
        Source: 28-pl
        Content: Kein Verzicht. Das Versäumnis oder die Verzögerung bei der Ausübung eines Rechts oder Rechtsmittels aus diesem Vertrag stellt keinen Verzicht auf dieses (oder ein anderes) Recht oder Rechtsmittel dar.\n\n11.7 Salvatorische Klausel. Die Ungültigkeit, Rechtswidrigkeit oder Unvollstreckbarkeit einer Bedingung (oder eines Teils einer Bedingung) dieses Vertrages beeinträchtigt nicht das Fortbestehen des Rests der Bedingung (falls vorhanden) und dieses Vertrages.\n\n11.8 Keine Agentur. Sofern nicht ausdrücklich anders angegeben, schafft dieser Vertrag keine Agentur, Partnerschaft oder Joint Venture jeglicher Art zwischen den Parteien.\n\n11.9 Keine Drittbegünstigten.
        Source: 30-pl
        Content: (b) wenn Google glaubt, in gutem Glauben, dass der Vertriebshändler gegen Anti-Korruptionsgesetze (wie in Klausel 8.5 definiert) verstoßen hat oder dass ein solcher Verstoß wahrscheinlich eintreten wird,
        Source: 4-pl
        =========
        ENDGÜLTIGE ANTWORT: Dieser Vertrag wird durch englisches Recht geregelt.
        SOURCES: 28-pl

        FRAGE: Was hat der Präsident über Michael Jackson gesagt?
        =========
        Content: Frau Sprecherin, Frau Vizepräsidentin, unsere First Lady und der zweite Gentleman. Mitglieder des Kongresses und des Kabinetts. Richter des Obersten Gerichtshofs. Meine amerikanischen Mitbürger. \n\nLetztes Jahr hat uns COVID-19 auseinandergebracht. In diesem Jahr sind wir endlich wieder zusammen. \n\nHeute Abend treffen wir uns als Demokraten, Republikaner und Unabhängige. Aber vor allem als Amerikaner. \n\nMit einer Pflicht zueinander, zum amerikanischen Volk, zur Verfassung. \n\nUnd mit der unerschütterlichen Entschlossenheit, dass die Freiheit immer über die Tyrannei siegen wird. \n\nVor sechs Tagen versuchte Russlands Wladimir Putin, die Grundlagen der freien Welt zu erschüttern, in der Hoffnung, sie könnte sich seinen bedrohlichen Methoden beugen. Aber er hat sich schwer verkalkuliert. \n\nEr dachte, er könnte in die Ukraine einrollen und die Welt würde sich umdrehen. Stattdessen traf er auf eine Mauer der Stärke, die er sich nie vorgestellt hatte. \n\nEr traf das ukrainische Volk. \n\nVon Präsident Selenskyj bis zu jedem Ukrainer, ihre Furchtlosigkeit, ihr Mut, ihre Entschlossenheit inspiriert die Welt. \n\nGruppen von Bürgern, die Panzer mit ihren Körpern blockieren. Jeder, von Studenten bis zu Rentnern, Lehrer, die zu Soldaten wurden, verteidigt ihre Heimat.
        Source: 0-pl
        Content: Und wir werden nicht aufhören. \n\nWir haben so viel an COVID-19 verloren. Zeit miteinander. Und am schlimmsten, so viel Verlust von Leben. \n\nNutzen wir diesen Moment zum Reset. Lasst uns aufhören, COVID-19 als parteipolitische Trennlinie zu sehen und es für das zu erkennen, was es ist: Eine schreckliche Krankheit. \n\nLasst uns aufhören, uns als Feinde zu sehen und anfangen, uns als das zu sehen, was wir wirklich sind: Amerikaner. \n\nWir können nicht ändern, wie gespalten wir gewesen sind. Aber wir können ändern, wie wir vorangehen - bei COVID-19 und anderen Fragen, die wir gemeinsam angehen müssen. \n\nVor kurzem besuchte ich das New Yorker Polizeidepartment Tage nach den Beerdigungen von Officer Wilbert Mora und seinem Partner, Officer Jason Rivera. \n\nSie reagierten auf einen 9-1-1 Anruf, als ein Mann sie mit einer gestohlenen Waffe erschoss und tötete. \n\nOfficer Mora war 27 Jahre alt. \n\nOfficer Rivera war 22. \n\nBeide dominikanische Amerikaner, die auf denselben Straßen aufwuchsen, die sie später als Polizisten patrouillierten. \n\nIch sprach mit ihren Familien und sagte ihnen, dass wir für ihr Opfer ewig in Schuld stehen und ihre Mission fortsetzen werden, das Vertrauen und die

        Sicherheit, die jede Gemeinschaft verdient, wiederherzustellen.
        Source: 24-pl
        Content: Und ein stolzes ukrainisches Volk, das 30 Jahre Unabhängigkeit gekannt hat, hat wiederholt gezeigt, dass es niemanden tolerieren wird, der versucht, ihr Land rückwärts zu nehmen. \n\nAn alle Amerikaner, ich werde ehrlich zu euch sein, wie ich es immer versprochen habe. Ein russischer Diktator, der ein fremdes Land überfällt, hat Kosten auf der ganzen Welt. \n\nUnd ich ergreife robuste Maßnahmen, um sicherzustellen, dass der Schmerz unserer Sanktionen auf die russische Wirtschaft abzielt. Und ich werde jedes Mittel in unserer Macht stehende nutzen, um amerikanische Unternehmen und Verbraucher zu schützen. \n\nHeute Abend kann ich ankündigen, dass die Vereinigten Staaten mit 30 anderen Ländern zusammengearbeitet haben, um 60 Millionen Barrel Öl aus Reserven auf der ganzen Welt freizugeben. \n\nAmerika wird diese Bemühungen anführen und 30 Millionen Barrel aus unserer eigenen strategischen Erdölreserve freigeben. Und wir sind bereit, bei Bedarf mehr zu tun, vereint mit unseren Verbündeten. \n\nDiese Schritte werden helfen, die Benzinpreise hier zu Hause abzuschwächen. Und ich weiß, die Nachrichten darüber, was passiert, können beunruhigend erscheinen. \n\nAber ich möchte, dass ihr wisst, dass wir okay sein werden.
        Source: 5-pl
        Content: Mehr Unterstützung für Patienten und Familien. \n\nUm dorthin zu gelangen, fordere ich den Kongress auf, ARPA-H, die Advanced Research Projects Agency for Health, zu finanzieren. \n\nEs basiert auf DARPA - dem Verteidigungsministerium-Projekt, das zum Internet, GPS und so vielem mehr führte. \n\nARPA-H wird einen einzigen Zweck haben - Durchbrüche bei Krebs, Alzheimer, Diabetes und mehr zu erzielen. \n\nEine Einheitsagenda für die Nation. \n\nWir können das schaffen. \n\nMeine amerikanischen Mitbürger - heute Abend haben wir uns in einem heiligen Raum versammelt - der Zitadelle unserer Demokratie. \n\nIn diesem Kapitol haben Generation um Generation Amerikaner große Fragen inmitten großer Konflikte diskutiert und Großes vollbracht. \n\nWir haben für die Freiheit gekämpft, die Freiheit erweitert, Totalitarismus und Terror besiegt. \n\nUnd die stärkste, freieste und wohlhabendste Nation aufgebaut, die die Welt je gekannt hat. \n\nJetzt ist die Stunde. \n\nUnser Moment der Verantwortung. \n\nUnser Test der Entschlossenheit und des Gewissens, der Geschichte selbst. \n\nIn diesem Moment wird unser Charakter geformt. Unser Zweck ist gefunden. Unsere Zukunft wird geschmiedet. \n\nNun, ich kenne diese Nation.
        Source: 34-pl
        =========
        ENDGÜLTIGE ANTWORT: Der Präsident hat Michael Jackson nicht erwähnt.
        QUELLEN:

        FRAGE: {question}
        =========
        {summaries}
        =========
        ENDGÜLTIGE ANTWORT:"""

        question_gen_prompt = """Angesichts der folgenden Konversation und einer anschliessenden Frage, formulieren Sie die Nachfrage so um, dass sie als eigenstaendige Frage gestellt werden kann.
            Alle Fragen und Antworten muessen auf Deutsch sein.
            Wenn Du die Antwort nicht kennst, sage einfach, dass Du es nicht weisst, versuche nicht, eine Antwort zu erfinden.

            Chatverlauf:
            {chat_history}
            Nachfrage: {question}
            Alle Fragen und Antworten muessen auf Deutsch sein.
            Eigenständige Frage:
            """

        GERMAN_QA_PROMPT = PromptTemplate(
            template=qa_prompt, input_variables=["summaries", "question"]
        )

        GERMAN_QG_PROMPT = PromptTemplate(
            template=question_gen_prompt, input_variables=["chat_history", "question"])

        GERMAN_DOC_PROMPT = PromptTemplate(
            template="Inhalt: {page_content}\nQuelle: {source}",
            input_variables=["page_content", "source"])
        
        llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k-0613')

        question_generator = LLMChain(llm=llm,prompt=GERMAN_QG_PROMPT)

        doc_chain = load_qa_with_sources_chain(llm,
                                            chain_type="stuff",
                                            prompt=GERMAN_QA_PROMPT,
                                            document_prompt=GERMAN_DOC_PROMPT
                                            )

        chain = ConversationalRetrievalChain(
            retriever=self.milvus.as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            return_generated_question=True
)


        result = chain({"question": query, "chat_history": history})

        return result