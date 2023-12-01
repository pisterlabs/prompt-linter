from coherence_interface.getAnaphoraReward import getAnaphoraReward
from coherence_interface.getAppositionReward import getAppositionReward

def getLocalReward(tok_sents):
    anaphora = getAnaphoraReward(tok_sents)
    apposition = getAppositionReward(tok_sents)
    return anaphora+apposition