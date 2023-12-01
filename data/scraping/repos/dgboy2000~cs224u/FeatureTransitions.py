from coherence.CreateTransitions import CreateTransitions
from FeatureBase import FeatureBase
import nltk
import numpy as np
import os
import pickle

class FeatureTransitions(object):

    word_splitter = nltk.WordPunctTokenizer()
    transitions = CreateTransitions().getTransitions()

    def __init__(self):
        self.features = np.array(())
        self.type = 'real'

    def numFeatures(self):
        """Return the number of features for this feature vector"""
        return length(self.features)

    def featureType(self):
        """Returns string description of the feature type, such as 'real-valued', 'binary', 'enum', etc."""
        return self.type

    def getFeatureMatrix(self):
        """Returns ordered list of features."""
        return self.features

    def extractFeatures(self, ds, corpus):
        """Extracts features from a DataSet ds"""
        lenfeats = list()
        for line in ds.getRawText():
            curfeat = list()
            words = FeatureTransitions.word_splitter.tokenize(line) 
            words = [word.lower() for word in words]
            words = [word for word in words if word.isalpha()]
            word_set = set(words)
            word_set.discard('')
            unique_transitions_count = 0
            transitions_count = 0 
            
            for word in words:
                if word in FeatureTransitions.transitions:
                    transitions_count += 1
            
            for word in word_set:
                if word in FeatureTransitions.transitions:
                    unique_transitions_count += 1
            
            #curfeat.append(transitions_count)
            curfeat.append(unique_transitions_count)
            #curfeat.append(unique_transitions_count/len(words))
            
            lenfeats.append(curfeat)

        self.features = np.asarray(lenfeats)
        return

FeatureBase.register(FeatureTransitions)

