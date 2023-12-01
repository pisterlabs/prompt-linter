import datasets

from raft_baselines.classifiers import CohereClassifier


train = datasets.load_dataset(
    "ought/raft", "neurips_impact_statement_risks", split="train"
)
classifier = CohereClassifier(
    train, config="neurips_impact_statement_risks", do_semantic_selection=False, #True
    num_prompt_training_examples=10,
    engine='baseline-shark'
)
# import ipdb; ipdb.set_trace()
print(classifier.classify({"Paper title": "Fast Facial Recognition", "Impact statement": "This technology could possibly used for oppression of minorities in authoritarian countries."}))
