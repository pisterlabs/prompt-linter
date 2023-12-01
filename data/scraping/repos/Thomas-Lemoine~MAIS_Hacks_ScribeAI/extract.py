import cohere
import re

API_KEY = "7YJur2HcZzSilwDv44ru6rLHCmVxiIhXC3nl2B1U"
co = cohere.Client(API_KEY)


def split_into_sentences(text):  # from https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|ca|us|tech|ai)"
    digits = "([0-9])"
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    if "..." in text:
        text = text.replace("...", "<prd><prd><prd>")
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms+" "+starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" "+suffixes+"[.] "+starters, " \\1<stop> \\2", text)
    text = re.sub(" "+suffixes+"[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def summarize(text, max_toks):
    prompt = f"{text}.\nIn summary,"
    prediction = co.generate(
        model='xlarge',
        prompt=prompt,
        return_likelihoods='GENERATION',
        stop_sequences=['.'],
        max_tokens=max_toks,
        temperature=0.0,
    )
    return prediction.generations[0].text


def separation(text):
    threshold = 500
    sentence_split = split_into_sentences(text)

    blk, blk_len, blk_lst = "", 0, []
    for sentence in sentence_split:
        sentence_len = co.tokenize(sentence).length
        if blk_len + sentence_len > threshold:
            blk_lst.append(blk)
            blk, blk_len = "", 0
        blk += sentence
        blk_len += sentence_len
    blk_lst.append(blk)
    return blk_lst


def terms(text, max_toks):
    prompt = f'''
    This is a passage followed by 2 important terms.
    
    Dinosaurs are a diverse group of reptiles of the clade Dinosauria. They first appeared during the Triassic period, between 243 and 233.23 million years ago (mya), although the exact origin and timing of the evolution of dinosaurs is the subject of active research. They became the dominant terrestrial vertebrates after the Triassic–Jurassic extinction event 201.3 mya; their dominance continued throughout the Jurassic and Cretaceous periods. The fossil record shows that birds are feathered dinosaurs, having evolved from earlier theropods during the Late Jurassic epoch, and are the only dinosaur lineage known to have survived the Cretaceous–Paleogene extinction event approximately 66 mya. Dinosaurs can therefore be divided into avian dinosaurs—birds—and paraphyletic pseudo-extinct non-avian dinosaurs, which are all dinosaurs other than birds. 
    2 Terms: avian dinosaurs, non-avian dinosaurs.
    
    Gradient descent is the process by which machines learn how to generate new faces, play hide and seek, and even beat the best humans at games like Dota. But what exactly is gradient descent? To answer that question, let's say you are trying to train your computer to listen to an audio file and recognize three spoken commands based on a label dataset. The first step is to formulate this machine learning task as a mathematical optimization problem. For example, you can work with a neural network whose weights are unknown variables, whose input is an audio data, and whose output is a vector of size 3 where each entry represents how much the neural network thinks the audio corresponds to each command. Then, for each example in the dataset, you compare the output of the neural network to the ideal output. Take the difference, the square and then the sum, and you get the cost of a single training example.
    2 Terms: gradient descent, optimization.
    
    Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.
    2 Terms: machine learning, artificial intelligence.
    
    Nietzsche's writing spans philosophical polemics, poetry, cultural criticism, and fiction while displaying a fondness for aphorism and irony. Prominent elements of his philosophy include his radical critique of truth in favor of perspectivism; a genealogical critique of religion and Christian morality and a related theory of master–slave morality; the aesthetic affirmation of life in response to both the "death of God" and the profound crisis of nihilism; the notion of Apollonian and Dionysian forces; and a characterization of the human subject as the expression of competing wills, collectively understood as the will to power. He also developed influential concepts such as the Übermensch and his doctrine of eternal return. In his later work, he became increasingly preoccupied with the creative powers of the individual to overcome cultural and moral mores in pursuit of new values and aesthetic health. His body of work touched a wide range of topics, including art, philology, history, music, religion, tragedy, culture, and science, and drew inspiration from Greek tragedy as well as figures such as Zoroaster, Arthur Schopenhauer, Ralph Waldo Emerson, Richard Wagner and Johann Wolfgang von Goethe. 
    2 Terms: Nietzsche, nihilism.
    
    Canada, the second largest country in the world, occupies most of the northern part of North America, covering the vast land area from the United States and the South to the Arctic Circle in the North. It is a country of enormous distances and rich natural resources. Indigenous people lived in what is now Canada for thousands of years before the first Europeans arrived. They are known as the First Nations and the Inuit People. The first European people to come to Canada arrived between 15,000 and 30,000 years ago across a land bridge that joined Asia and North America. Around 1000 AD, the Viking explorer, Lee Farrickson, reached Newfoundland, Canada. He tried to establish a settlement, but it didn't last. In the early 16th century, Europeans started exploring Canada's eastern coast, beginning with John Cabot from England. Between 1534 and 1542, Jacques Cartier made three voyages across the Atlantic, claiming the land for King Francis I of France. Cartier had two captured guides, speak the Iroquoisan word canata, meaning village. By the 1550s, the name of Canada began appearing on maps. Parts of Canada were settled by France and parts by Great Britain. In 1605, Port Royal was built in Acadia by the French, led by Samuel de Champlain, and three years later, he started settling Quebec.
    2 Terms: Canada, indigenous people.
    
    {text}
    2 Terms: '''
    prediction = co.generate(
        model='xlarge',
        prompt=prompt,
        return_likelihoods = 'GENERATION',
        stop_sequences=["."],
        max_tokens=max_toks,
        temperature=0.2,
        frequency_penalty=0.4,
    )
    return prediction.generations[0].text


def descriptions(text, term, max_toks):
    prompt = f'''
    This is a passage followed by a question on an important term and description of what it is.
    
    Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.
    What is training data? sample data that Machine learning algorithms build a model on
    
    Dinosaurs are a diverse group of reptiles of the clade Dinosauria. They first appeared during the Triassic period, between 243 and 233.23 million years ago (mya), although the exact origin and timing of the evolution of dinosaurs is the subject of active research. They became the dominant terrestrial vertebrates after the Triassic–Jurassic extinction event 201.3 mya; their dominance continued throughout the Jurassic and Cretaceous periods. The fossil record shows that birds are feathered dinosaurs, having evolved from earlier theropods during the Late Jurassic epoch, and are the only dinosaur lineage known to have survived the Cretaceous–Paleogene extinction event approximately 66 mya. Dinosaurs can therefore be divided into avian dinosaurs—birds—and paraphyletic pseudo-extinct non-avian dinosaurs, which are all dinosaurs other than birds. 
    What is avian dinosaurs? birds or feathered dinosaurs that having evolved from earlier theropods during the Late Jurassic epoch
    
    Dinosaurs are a diverse group of reptiles of the clade Dinosauria. They first appeared during the Triassic period, between 243 and 233.23 million years ago (mya), although the exact origin and timing of the evolution of dinosaurs is the subject of active research. They became the dominant terrestrial vertebrates after the Triassic–Jurassic extinction event 201.3 mya; their dominance continued throughout the Jurassic and Cretaceous periods. The fossil record shows that birds are feathered dinosaurs, having evolved from earlier theropods during the Late Jurassic epoch, and are the only dinosaur lineage known to have survived the Cretaceous–Paleogene extinction event approximately 66 mya. Dinosaurs can therefore be divided into avian dinosaurs—birds—and paraphyletic pseudo-extinct non-avian dinosaurs, which are all dinosaurs other than birds. 
    What is non-avian dinosaurs? paraphyletic pseudo-extinct that are not birds
    
    Nietzsche's writing spans philosophical polemics, poetry, cultural criticism, and fiction while displaying a fondness for aphorism and irony. Prominent elements of his philosophy include his radical critique of truth in favor of perspectivism; a genealogical critique of religion and Christian morality and a related theory of master–slave morality; the aesthetic affirmation of life in response to both the "death of God" and the profound crisis of nihilism; the notion of Apollonian and Dionysian forces; and a characterization of the human subject as the expression of competing wills, collectively understood as the will to power. He also developed influential concepts such as the Übermensch and his doctrine of eternal return. In his later work, he became increasingly preoccupied with the creative powers of the individual to overcome cultural and moral mores in pursuit of new values and aesthetic health. His body of work touched a wide range of topics, including art, philology, history, music, religion, tragedy, culture, and science, and drew inspiration from Greek tragedy as well as figures such as Zoroaster, Arthur Schopenhauer, Ralph Waldo Emerson, Richard Wagner and Johann Wolfgang von Goethe. 
    What is Nietzche? a philosopher and cultural critic who was fond for aphorism and irony and would critique truth in favor perspectivism, affirm life in response to the "death of God" and nihilism, and characterizing humans as competing wills.
    
    In logic and theoretical computer science, and specifically proof theory and computational complexity theory, proof complexity is the field aiming to understand and analyse the computational resources that are required to prove or refute statements. Research in proof complexity is predominantly concerned with proving proof-length lower and upper bounds in various propositional proof systems. For example, among the major challenges of proof complexity is showing that the Frege system, the usual propositional calculus, does not admit polynomial-size proofs of all tautologies. Here the size of the proof is simply the number of symbols in it, and a proof is said to be of polynomial size if it is polynomial in the size of the tautology it proves. 
    What is proof complexity? the field aiming to understand and analyse the computational resources that are required to prove or refute statements
    
    In logic and theoretical computer science, and specifically proof theory and computational complexity theory, proof complexity is the field aiming to understand and analyse the computational resources that are required to prove or refute statements. Research in proof complexity is predominantly concerned with proving proof-length lower and upper bounds in various propositional proof systems. For example, among the major challenges of proof complexity is showing that the Frege system, the usual propositional calculus, does not admit polynomial-size proofs of all tautologies. Here the size of the proof is simply the number of symbols in it, and a proof is said to be of polynomial size if it is polynomial in the size of the tautology it proves. 
    What is the size of a proof? the number of symbols in it

    {text}
    What is {term}?'''
    prediction = co.generate(
        model='xlarge',
        prompt=prompt,
        return_likelihoods = 'GENERATION',
        stop_sequences=["."],
        max_tokens=max_toks,
        temperature=0.2,
    )
    return prediction.generations[0].text


def generate_all(text, compression):
    all_summary, all_term, all_desc = "", [], []
    last_two_summary = ""
    blk_list = separation(text)
    cnt = 0
    for b in blk_list:
        s = summarize(b, co.tokenize(b).length // compression)
        all_summary += s
        last_two_summary += s
        if cnt % 2 == 1:
            terms_split = terms(last_two_summary, 100).split(",")
            all_term += terms_split
            desc = [descriptions(last_two_summary, i, 100) for i in terms_split]
            all_desc += desc
            # print(terms_split)
            # print(desc)
            last_two_summary = ""
        cnt += 1
    return all_summary, all_term, all_desc


if __name__ == "__main__":
    wiki = '''Machine learning From Wikipedia, the free encyclopedia Jump to navigation Jump to search For the journal, see Machine Learning (journal)."Statistical learning" redirects here.For statistical learning in linguistics, see statistical learning in language acquisition.Part of a series on Machine learning and data mining Scatterplot featuring a linear support vector machine's decision boundary (dashed line) Problems Supervised learning (classification • regression) Clustering Dimensionality reduction Structured prediction Anomaly detection Artificial neural network Reinforcement learning Learning with humans Model diagnostics Theory Machine-learning venues Related articles      vte  Part of a series on Artificial intelligence Anatomy-1751201 1280.png Major goals      Artificial general intelligence Planning Computer vision General game playing Knowledge reasoning Machine learning Natural language processing Robotics  Approaches Philosophy History Technology Glossary      vte  Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.[1] It is seen as a part of artificial intelligence.Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.[2] Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.[3]  A subset of machine learning is closely related to computational statistics, which focuses on making predictions using computers, but not all machine learning is statistical learning.The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning.[5][6] Some implementations of machine learning use data and neural networks in a way that mimics the working of a biological brain.[7][8] In its application across business problems, machine learning is also referred to as predictive analytics.'''
    print(generate_all(wiki, 8))
