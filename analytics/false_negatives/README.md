# Detect False Positives and False Negatives in Prompt Detection

### Note: This directory was initially created to collect prompt data from GitHub repos. However, that task was moved to the gen_prompts directory. This directory is now used to detect false positives and false negatives in prompt detection.

Dir notes:
- /.build and ./vendor contains tree-sitter dependencies
- ./flair_corpus and prompt_classification.csv contain data to train the prompt classifiers
- ./resources contains the fine-tuned text classifier model (trained on flair_corpus)
- ./repos contains the repos that we downloaded from GitHub with 4 or more stars (more details in ../data)

File notes:
- parser.ipynb downloads all repo files from GitHub and parses them to collect prompts
- prompt_classification.ipynb trains a text classifiers to classify strings as prompts (2 different classifiers)
- (**DEPRECATED**) parser_comparison.ipynb compares the prompts collected by the default parsing heuristic 
    (checking for '\n' in strings assigned to variables) to the prompts collected by the text classifiers
    - Generates different results for the number of repos with high prompt-densities. 
    (Something to consider for the future)
    - Downloads 0 star repos and applies the 3 parsers on them to compare the results
- (**DEPRECATED**) log_classifier.pkl is the log classifier trained on prompt_classification.csv (using k-fold cross validation)
- prompt_classifications.csv is the data (based only on repos with >=4 stars) used to train the prompt classifiers
