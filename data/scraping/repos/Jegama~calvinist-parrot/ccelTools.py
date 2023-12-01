from langchain.tools import Tool
import requests, string

def formater(output):
    sources = {}
    alphabet = string.ascii_lowercase
    a = 0

    for source in output['source_nodes']:
        if source['node']['metadata']['title'] not in sources.keys():
            sources[source['node']['metadata']['title']] = {'authors': source['node']['metadata']['authors'], 'score': [f"{alphabet[a]}. {round(source['score'], 3)}"]}
        else:
            sources[source['node']['metadata']['title']]['score'].append(f"{alphabet[a]}. {round(source['score'], 3)}")
        a += 1

    source_text = "Sources:\n"
    n = 1

    for source in sources.keys():
        source_text += f"{n}. {source} by {sources[source]['authors']} - Confidence: {', '.join(sources[source]['score'])}\n"
        n += 1

    return source_text

def early_christian_literature(question):
    response = requests.post('https://early-christian-literature-west2-e4y6sp3yrq-wl.a.run.app/query', json={'question': question})
    output = response.json()
    return output['response'] + '\n\n' + formater(output)

def reformed_commentaries(question):
    response = requests.post('https://reformed-commentaries-west2-e4y6sp3yrq-wl.a.run.app/query', json={'question': question})
    output = response.json()
    return output['response'] + '\n\n' + formater(output)

def systematic_theology(question):
    response = requests.post('https://systematic-theology-west2-e4y6sp3yrq-wl.a.run.app/query', json={'question': question})
    output = response.json()
    return output['response'] + '\n\n' + formater(output)

toolkit = [
    Tool(
        name="Early Christian Literature", 
        func=early_christian_literature, 
        description="Early texts, letters, and documents from the initial centuries of Christianity. - Includes books like ANF02. Fathers of the Second Century: Hermas, Tatian, Athenagoras, Theophilus, and Clement of Alexandria (Entire), ANF04. Fathers of the Third Century: Tertullian, Part Fourth; Minucius Felix; Commodian; Origen, Parts First and Second, ANF06. Fathers of the Third Century: Gregory Thaumaturgus, Dionysius the Great, Julius Africanus, Anatolius, and Minor Writers, Methodius, Arnobius and authors like Schaff, Philip (1819-1893) (Editor), Lightfoot, Joseph Barber (1828-1889), Pearse, Roger."
    ),
    Tool(
        name="Reformed Commentaries", 
        func=reformed_commentaries, 
        description="Reformed books focusing on the interpretation, analysis, and study of biblical texts. - Includes books like Harmony of the Law - Volume 3, Preface to the Letter of St. Paul to the Romans, Why Four Gospels? and authors like Pink, Arthur W., Calvin, Jean, Calvin, John (1509 - 1564)."
    ),
    Tool(
        name="Systematic Theology", 
        func=systematic_theology, 
        description="Comprehensive exploration of Christian doctrines and theology. - Includes books like A Body of Practical Divinity, Doctrinal Theology, History of Dogma - Volume IV and authors like Hodge, Charles (1797-1878), Hopkins, Samuel (1721-1803), Gill, John (1697-1771)."
    ),
]