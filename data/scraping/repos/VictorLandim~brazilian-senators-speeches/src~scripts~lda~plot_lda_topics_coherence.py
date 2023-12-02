import pickle
from gensim.models.wrappers import LdaMallet
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary


def run():
    lda_path = 'src/data/models/50_it_10000/model'
    lda = LdaMallet.load(lda_path)

    # dictionary = pickle.load(open("src/data/dictionary.pickle", "rb"))
    # texts = [speech['text_preproc'] for speech in pickle.load(
    #     open("src/data/speeches_all_preproc_filtered.pickle", "rb"))]
    # dictionary = Dictionary(texts)

    # coherence_model = CoherenceModel(
    #     model=lda,
    #     texts=texts,
    #     dictionary=dictionary,
    #     coherence='c_npmi'
    # )

    # print(coherence_model.get_coherence_per_topic())

    vals = [-0.10828429828232149, -0.12697691840221148, -0.08875932386915583, -0.11614475256298387, -0.12032688203574213, -0.04719309266492472, -0.1540929051289418, -0.0497353195084467, -0.09272850814857891, -0.12472646026329233, -0.08703374612179167, -0.06561583171539828, -0.1290658619970064, -0.06922417098693691, -0.1374746321391089, -0.06738518155673777, -0.05456845624365386, -0.0922529599593146, -0.08548265734877294, -0.07461416807447167, -0.08221879178327315, -0.03428110087530537, -0.07502422647248216, -0.10447097742063385, -0.10127995373724456, -
            0.13662571173567714, -0.09010722910383585, -0.1418273092412291, -0.09287703206979078, -0.08528880371161812, -0.07604214144141853, -0.1262384388820645, -0.07079305272496314, -0.11903393245341251, -0.0691494779575121, -0.11788051875353897, -0.1291262446951616, -0.11912889825650654, -0.1381799645911901, -0.057929748979572725, -0.0956934043056181, -0.12517972261730606, -0.07957637053356044, -0.10899175009232531, -0.11156706125198546, -0.03758538257177168, -0.06468412545115033, -0.08016096605731907, -0.05922196454884588, -0.10615329217165044]

    print(len(vals))

    topics = [
        (topic,
         vals[topic],
         " ".join([word for word, weight in words]),
         )
        for (topic, words) in lda.show_topics(num_topics=lda.num_topics, formatted=False)
    ]

    for t in sorted(topics, key=lambda x: x[1], reverse=True):
        print(t)


if __name__ == '__main__':
    run()
