# Parsing All Repo Files to collect Prompts

Dir notes:
- /.build and ./vendor contains tree-sitter dependencies
- ./flair_corpus and prompt_classification.csv contain data to train the prompt classifiers
- ./resources contains the fine-tuned text classifier model (trained on flair_corpus)
- ./repos contains the repos that we downloaded from GitHub with 4 or more stars (more details in ../data)

File notes:
- parser.ipynb downloads all repo files from GitHub and parses them to collect prompts
- prompt_classification.ipynb trains a text classifiers to classify strings as prompts (2 different classifiers)
- parser_comparison.ipynb compares the prompts collected by the default parsing heuristic 
    (checking for '\n' in strings assigned to variables) to the prompts collected by the text classifiers
    - Generates different results for the number of repos with high prompt-densities. 
    (Something to consider for the future)

TODOs:
- Consult with Professor to decide on strategy for selecting high prompt-density repos
- Decide if we also want to download repos with less than 4 stars (currently we only have repos with >=4 stars)