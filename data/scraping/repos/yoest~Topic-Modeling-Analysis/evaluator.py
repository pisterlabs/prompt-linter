from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

import pandas as pd

from scipy.optimize import linear_sum_assignment


class Evaluator:

    def __init__(self, model_output: dict) -> None:
        """ Can be use to evaluate Topic Modeling models.

        Args:
            model_output (dict): The output of a trained model. This should be on the OCTIS format (see https://github.com/MIND-Lab/OCTIS).
        """
        self.model_output = model_output

    def compute_coherence(self) -> dict:
        """ Compute the coherence of the model.

        Returns:
            dict: A dictionary containing the coherence of the model (using the 'c_v', 'c_uci', 'c_npmi' and 'u_mass' measures).
        """
        c_v = Coherence(measure='c_v')
        c_uci = Coherence(measure='c_uci')
        c_npmi = Coherence(measure='c_npmi')
        u_mass = Coherence(measure='u_mass')

        return {
            'c_v': c_v.score(self.model_output),
            'c_uci': c_uci.score(self.model_output),
            'c_npmi': c_npmi.score(self.model_output),
            'u_mass': u_mass.score(self.model_output)
        }
    
    def compute_diversity(self) -> float:
        """ Compute the diversity of the model.

        Returns:
            float: The diversity of the model.
        """
        diversity = TopicDiversity()
        return diversity.score(self.model_output)
    
    def __compute_similarity(self, true_words: list, extracted_words: list) -> float:
        """ Compute the similarity between two topics

        Args:
            true_words (list): The words of the true topic
            extracted_words (list): The words of the extracted topic

        Returns:
            float: The similarity between the two topics
        """
        similarity = 0
        total_count = 0

        for extracted_word, extracted_word_count in extracted_words:
            same_word_true_count = 0
            for true_word, true_word_count in true_words:
                if true_word == extracted_word:
                    same_word_true_count = true_word_count
                    break

            current_similarity = 1 - abs((extracted_word_count - same_word_true_count) / max(extracted_word_count, same_word_true_count))
            current_similarity *= (extracted_word_count + same_word_true_count)
            similarity += current_similarity

            total_count += extracted_word_count + same_word_true_count

        return similarity / total_count
    
    def __compute_similarity_matrix(self, words_by_extracted_topics: dict, words_by_true_topics: dict) -> list:
        """ Compute the similarity matrix between each topic

        Args:
            words_by_extracted_topics (dict): The words for each extracted topic
            words_by_true_topics (dict): The words for each true topic

        Returns:
            list: The similarity matrix
        """
        similarity_matrix = []
        for _, extracted_topic in enumerate(words_by_extracted_topics):
            row = []
            for _, true_topic in enumerate(words_by_true_topics):
                row.append(self.__compute_similarity(words_by_true_topics[true_topic], words_by_extracted_topics[extracted_topic]))
            similarity_matrix.append(row)

        return similarity_matrix
    
    def compute_supervised_correlation(self, words_by_extracted_topics: dict, words_by_class: dict) -> float:
        """ Compute the supervised correlation between the words of the extracted topics and the words of the classes.

        Args:
            words_by_extracted_topics (dict): The words of the extracted topics.
            words_by_class (dict): The words of the classes.

        Returns:
            float: The supervised correlation between the words of the extracted topics and the words of the classes.
        """
        new_words_by_extracted_topics = {}
        for key in words_by_extracted_topics:
            new_words_by_extracted_topics[key] = list(words_by_extracted_topics[key].items())

        new_words_by_class = {}
        for key in words_by_class:
            new_words_by_class[key] = list(words_by_class[key].items())

        similarity_matrix = self.__compute_similarity_matrix(new_words_by_extracted_topics, new_words_by_class)

        row_indices, col_indices = linear_sum_assignment(similarity_matrix, maximize=True)

        true_topic_labels = list(new_words_by_class.keys())

        assignment_similarity = {}

        for extracted_topic_idx, true_topic_idx in zip(row_indices, col_indices):
            assignment_similarity[true_topic_labels[true_topic_idx]] = similarity_matrix[true_topic_idx][extracted_topic_idx]

        supervised_correlation = sum(assignment_similarity.values()) / len(assignment_similarity.values())
        return supervised_correlation

        
    def __compute_single_correlation(self, words_for_extracted_topic: dict, words_for_class: dict) -> float:
        """ Compute the correlation between the words of a topic and the words of a class.

        Args:
            words_for_extracted_topic (dict): The words of the topic.
            words_for_class (dict): The words of the class.

        Returns:
            float: The correlation between the words of a topic and the words of a class.
        """
        resulting_score = 0
        total_count = 0

        for topic_word in words_for_extracted_topic:
            count_topic_word = words_for_extracted_topic[topic_word]
            count_class_word = words_for_class.get(topic_word, 0)
            total_count += count_topic_word + count_class_word

            score = 1 - abs((count_topic_word - count_class_word) / max(count_topic_word, count_class_word))
            score = score * (count_topic_word + count_class_word)

            resulting_score += score

        return resulting_score / total_count