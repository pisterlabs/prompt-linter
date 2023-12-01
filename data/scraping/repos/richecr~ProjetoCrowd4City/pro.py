import os
import re
import csv
import nltk
import spacy
import gensim
import scipy.sparse
import pandas as pd

from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess, lemmatize
from gensim.models import LdaModel, LdaMulticore
from smart_open import smart_open
from gensim import matutils, models
from gensim import corpora
from pprint import pprint

nltk.download('stopwords')
stop_words = stopwords.words('portuguese')

stop_words = stop_words + ["vez", "vem", "olha", "pessoal", "tudo", "dia", "aqui", "gente", "tá", "né", "calendário", "jpb", "agora", "voltar", "lá", "hoje", "aí", "ainda", "então", "vai", "porque", "moradores", "fazer", "rua", "bairro", "prefeitura", "todo", "vamos", "problema", "fica", "ver", "tô"]

### Carregando dados.
data = pd.read_csv('../textos_videos.csv', encoding='utf-8')
t = data['texto']

textos = []
for texto in t:
    textos.append(texto.lower())
# print(textos)

nlp = spacy.load('pt_core_news_sm')
data_processada = []

def buscar_entidade(palavra, entidades):
    for ent in entidades:
        if (ent.text == palavra):
            return ent
    return -1

allowed_postags = ['NOUN', 'ADJ', 'PRON']
for texto in textos:
    doc_out = []
    doc = nlp(texto)
    for token in doc:
        if (token.text not in stop_words):
            if (token.pos_ in allowed_postags):
                doc_out.append(token.text)
            else:
                continue
        else:
            continue
    data_processada.append(doc_out)

def processa(text):
    doc = nlp(texto)
    doc_out = []
    for token in doc:
        if (token.text not in stop_words):
            if (token.pos_ in allowed_postags):
                doc_out.append(token.text)
            else:
                continue
        else:
            continue
    return doc_out

# print(data_processada[0][:5])

dct = corpora.Dictionary(data_processada)

t = data_processada
corpus = [dct.doc2bow(line) for line in t]

lda_model = LdaModel(corpus=corpus,
                    id2word=dct,
                    num_topics=5, 
                    random_state=100,
                    update_every=1,
                    passes=80,
                    alpha='asymmetric',
                    per_word_topics=True)

topics = lda_model.print_topics(-1)
for topic in topics:
    print(topic)

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_processada, corpus=corpus, dictionary=dct, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score LDAModel: ', coherence_lda)


print("-------")
unseen_document = 'calendário JPB aqui nas nossas telas nós vamos agora até o bairro Jardim Paulistano zona sul de Campinas Você lembra que nossa equipe ouviu os moradores da Rua Riachuelo que reclamavam da falta de calçamento no local então o problema foi resolvido só que na época a prefeitura também se comprometeu e fazer o calçamento da Rua Ariel que fica bem pertinho essa parte foi feita mas só que pela metade Laisa grisi foi conferido calendário JPB desembarcou aqui no Jardim Paulistano E olha que maravilha hoje é possível andar na rua com calçamento sem tanta poeira sem pisar em lama Quando chove essa foi uma conquista dos moradores junto com calendário Desde o ano passado em 2015 quando a prefeitura calçou essa rua calça com a Rua Riachuelo também mas presta atenção dois passos seguintes e rua de terra essa rua que esse trechinho não foi calçado vou aqui conversar com os moradores já tá todo mundo reunido Por que me explica como é que pode só esse trechinho não foi calçada só esse trecho você imagina que fizeram as duas por duas partes né fizeram aquela parte de lá aí ficou a metade depois fizeram essa daqui aí deixar essa parte aqui sem sem tá feita né nessa parte de baixo é pior ainda porque quando chove a água invade a Casa dos moradores e olha só aqui nessa casa foi colocado um monte de pedra bem na frente para impedir que a água entre vamos lá falar com ela é dona Severina é dona Bill Olá tudo bom com a senhora como é que tá aqui essa situação a senhora Teve que colocar pedra aqui né é chover em entrar aqui sozinha imagina aperreio Aí tem que dar um jeito aqui é pior difícil hein dona Bill quanto tempo já que a senhora mora aqui nessa rua 8 anos viu o resultado de vergonha né a gente não tem né É porque se ele tivesse vergonha ele já tinha feito isso todos vocês moram aqui nessa rua aí o que que acontece nessas ruas aqui né aí o que que acontece a Rua Areal lá em cima Foi calçada a Rua Riachuelo também E vocês ficaram só um gostinho só na saudade e o pior que não se desviar da Lama dos buracos e ele prometeu Então olha você tá vendo aquela cerâmica Vale Aí depois ele dá o que é o povo que bota para que ele possa passar infelizmente é uma situação difícil a gente já pediu muitas vezes recado dado essa essa rua que já é assunto do calendário a gente conseguiu algumas ruas outras não voltamos em 2016 em 2016 o secretário André agra secretário de obras de Campina Grande e disse que ia voltar aqui não foi então vamos lá calendário novo quem é o representante'
bow_vector = dct.doc2bow(processa(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print(score, lda_model.print_topic(index, 10))

'''
print("-------")
unseen_document = 'Rua Rubens Coelho Pereira Filho nuclear para tecido calçada mais a placa com os dados da obra a gente simplesmente sumiu e o calçamento não chegou para completar uma empresa privada abrir um buraco no meio da rua e já caiu gente até roda de carro os moradores procuraram a prefeitura mais até agora nada mudou e ele só tiveram uma alternativa pediram ajuda ao calendário JPB a Rua Rubens Coelho Pereira Filho aqui no bairro do Cuiá está em obras agora É uma pena que essa obra que não tem nada a ver com a necessidade e com a reclamação da rua inteira que como a gente vai ver olha só não tem calçamento e também não tem saneamento básico são mais de 15 anos que os moradores tem muitos transtornos Principalmente nos períodos de chuva e a gente vai começar a conversar com alguns deles para falar de como é viver nessa situação seu Matusalém muito sofrimento aqui bom dia bom dia a todos e muito sofrimento a 14 anos que eu moro aqui e nenhuma infraestrutura foi foi feita aqui no Parque da prefeitura essa empresa veio para fazer a obra ela tá passando uns fazendo saneamento mas é de um condomínio particular passou pela rua de vocês é isso exatamente que a gente não tem nada não vai se beneficiar e nada do que essa consultora está fazendo inclusive teve transtorno para gravar afundando a gente tem foto muito transtorno carro afundando pessoas aqui já caíram aqui dentro do buraco já saíram todas arranhadas em Popular né sem comunicação sem nada a nossas calçadas foram invadidas com barro e também muita lama por causa da obra que passou aqui e também da chuva que é um pouco em madeirado vai ser descendo água estão aparecendo E aí você imagina era só no meio de todo esse material para pessoa está caminhando criança e doso para daqui não tem jeito se a luz é difícil andar nessa rua os buracos demais e a gente tem eu tenho dificuldade de locomover para continuo no caminhar na rua fico mais dentro de casa com medo de sair na rua com medo de cair de acontecer o pior comigo eu sair agora da Rua Rubens Coelho Pereira Filho que é aquela lá para vir aqui na Rita Carneiro porque que eu tô aqui nesse local tem uma placa que fala da pavimentação da Rua Rita carneiro e também um trecho da Rubens Coelho Pereira Filho mas segundo os moradores tá aqui o Fernando para falar alguns anos a pavimentação que tinha nessa placa ou numa placa parecida era só da Rubens Coelho e não chegou de forma alguma lá foi É isso mesmo uma placa anterior ela é indicada que seria metade da Rua Rubens Coelho pelo qual motivo eu não sei mesmo moradores foi modificado essa placa com outro valor e a nossa rua até hoje está aí do jeito que você mostrou na sua reportagem E aí eu vi a conclusão da pavimentação da Rita Carneiro'
bow_vector = dct.doc2bow(processa(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 10)))

print("-------")
unseen_document = "Jaguaribe a Escola Estadual Professora Maria do Carmo de Miranda já mudou muito desde a chegada do calendário por lá mas falta resolver um pequeno detalhe mas os estudantes não poderão utilizar é por isso que o calendário JPB continua aqui na Escola Estadual Professora Maria do Carmo de Miranda mesmo depois da obra de reforma tem sido concluída a gente precisa ter a garantia que esse espaço vai ser utilizado Falta muito pouco mas essa história ainda não acabou apresento para vocês agora o laboratório de biologia aqui eu tô vendo que tem aquelas maquetes do corpo humano do outro lado tem microscópios e outros equipamentos as outras estantes também estão cheias de máquinas todo esse material na nossa última visita aqui no início do mês de março tava encaixotado uma poeira danada e eu tô vendo que agora tá tudo no seu devido lugar aparentemente pronto para usar tá tudo certo para aula agora até agora ainda não foi inaugurado né você não tem fé que Vamos inaugurar atualizado É verdade que nem os professores ainda foram apresentados É verdade aos Laboratórios eles ficaram surpresos né porque antes não era assim Foi de repente ficou tudo arrumadinho e os professores não tinha nem noção dos equipamentos dos laboratórios tá todo mundo conhecendo hoje o novo laboratório de biologia né porque agora vai dar para chamar de laboratório né quando a gente entra a gente já tem a sensação de que está mesmo no laboratório que que tá faltando para que os estudantes possam utilizar esse espaço falta de instalações do ar condicionado só falta ar condicionado para funcionar e futuramente né daqui daqui uns dias nós estamos recebendo o quê da robótica os professores já vão fazer uma formação para atuar nessa área inclusive Nós temos dois alunos que foram representar Paraíba no robótica na China e a gente sabe que essa escola tem muito potencial da área eles vão ser monitores essa escola realmente tem muitos talentos item acima de tudo né estudantes interessados em usar esses esses passos para aprender Então hoje dia 14 de abril o carimbo Ainda é em andamento nesse segundo a secretaria de educação do estado esses ar-condicionados daqui dos laboratórios vão ser instalados no prazo de 15 dias com 15 dias de instalação mas 15 dias Os estudantes de gestão por aqui tá tudo certo certo dia 14 de Maio cai no sábado né o próximo dia útil seria o dia 16 mas aí a Paloma que começou essa história não vai poder estar aqui no dia 16 só no dia 19 essa data diretora do dia 19 Fica boa para senhora fica"
bow_vector = dct.doc2bow(processa(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 10)))


print("------------------------")
mallet_path = '/home/rick/Documentos/mallet-2.0.8/bin/mallet'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=5, id2word=dct)
# print(ldamallet.show_topics(formatted=False))
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_processada, dictionary=dct, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score LDAMallet: ', coherence_ldamallet)
topics = ldamallet.print_topics(-1)
for topic in topics:
    print(topic)

'''
'''
(0, [('situação', 0.021352313167259787), ('nada', 0.017334404775571115), ('lixo', 0.01658822178854322), ('sabe', 0.013947881988290667), ('passar', 0.013603489840431637), ('carro', 0.013086901618643095), ('ninguém', 0.011938927792446333), ('vezes', 0.011192744805418436), ('faz', 0.010389163127080702), ('dá', 0.008839398461715072)])
(1, [('obra', 0.04798346553457714), ('comunidade', 0.01653446542285778), ('prazo', 0.010780918333147134), ('parte', 0.009998882806390347), ('data', 0.009552005362529327), ('mês', 0.009160987599150933), ('resposta', 0.008714110155289912), ('secretário', 0.008602390794324656), ('secretaria', 0.008099653669981008), ('dar', 0.007932074628533126)])
(2, [('escola', 0.026628748707342297), ('resolvido', 0.01809720785935884), ('ano', 0.011504653567735263), ('local', 0.00969493278179938), ('boa', 0.008725439503619441), ('ficou', 0.00814374353671148), ('muro', 0.007885211995863495), ('chegou', 0.007562047569803516), ('grande', 0.007303516028955533), ('lado', 0.006851085832471561)])
(3, [('água', 0.0330845624963272), ('casa', 0.030498912851854028), ('esgoto', 0.01974496092143151), ('situação', 0.015102544514309221), ('cagepa', 0.014044778750661104), ('chuva', 0.013986013986013986), ('resolver', 0.012928248222365869), ('buraco', 0.012340600575894693), ('tava', 0.01098901098901099), ('resolvido', 0.010636422401128283)])
(4, [('praça', 0.02206913369505578), ('coisa', 0.014265683106748766), ('tava', 0.013595074071816132), ('comunidade', 0.013107358410046942), ('vendo', 0.011766140340181674), ('quadra', 0.010790709016643297), ('certeza', 0.009876242150826069), ('fazendo', 0.009754313235383772), ('bom', 0.008839846369566542), ('ginásio', 0.008413095165518502)])

(0, '0.018*"obra" + 0.008*"situação" + 0.007*"comunidade" + 0.006*"calçamento" + 0.005*"serviço" + 0.005*"nada" + 0.005*"volta" + 0.005*"mês" + 0.005*"tava" + 0.004*"mercado"')
(1, '0.014*"escola" + 0.008*"resolvido" + 0.008*"comunidade" + 0.007*"nada" + 0.006*"lixo" + 0.006*"tava" + 0.005*"obra" + 0.005*"situação" + 0.005*"serviço" + 0.005*"muro"')
(2, '0.022*"água" + 0.015*"casa" + 0.013*"esgoto" + 0.011*"situação" + 0.007*"cagepa" + 0.007*"buraco" + 0.007*"chuva" + 0.006*"obra" + 0.005*"carro" + 0.005*"calçamento"')
(3, '0.017*"praça" + 0.013*"obra" + 0.009*"comunidade" + 0.005*"tempo" + 0.005*"nada" + 0.005*"tava" + 0.004*"sabe" + 0.004*"dá" + 0.004*"vendo" + 0.004*"resolvido"')
(4, '0.015*"escola" + 0.010*"quadra" + 0.006*"situação" + 0.006*"reforma" + 0.006*"comunidade" + 0.006*"ano" + 0.006*"tava" + 0.006*"ginásio" + 0.005*"coisa" + 0.005*"obra"')
'''