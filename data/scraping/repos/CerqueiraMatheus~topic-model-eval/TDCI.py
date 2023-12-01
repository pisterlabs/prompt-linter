from octis.evaluation_metrics.metrics import AbstractMetric
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

class TDCI(AbstractMetric):
    def __init__(self, texts=None, topk=10, measure="c_npmi"):
        """
        Initialize metric

        Parameters
        ----------
        texts : list of documents (list of lists of strings)
        topk : how many most likely words to consider in
        the evaluation
        measure : (default 'c_npmi') measure to use inside .
        other measures: 'u_mass', 'c_v', 'c_uci', 'c_npmi'
        """
        super().__init__()

        self.tc = Coherence(texts=texts, topk=topk, measure=measure)
        self.td = TopicDiversity(topk=topk)

    def score(self, model_output):
        """
        Retrieve the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        score : TDCI score
        """

        a = (self.tc.score(model_output) + 1) / 2
        b = self.td.score(model_output)

        if (a == 0) and (b == 0):
            return 0

        return 2 * ((a * b) / (a + b))
