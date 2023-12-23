# DevGPT Dataset

Contains all chats from the DevGPT dataset in chats.json. The dataset is available at: https://github.com/NAIST-SE/DevGPT

## Dataset Structure
{
    "chat_url": {
        "url": "...",
        ...,
        "Status": 200,
        "NumberOfPrompts": 1,
        "TokensOfPrompts": 199,
        "TokensOfAnswers": 404,
        "Conversations": [
            {
                "Prompt": "...",
                "Answer": "..."
            },
            ...
        ]
    },
    ...
}

Some Notes: 
- devGPT_prompts.ipynb: creates chats.json from the DevGPT dataset from DevGPT's Zenodo repo at https://zenodo.org/records/8304091, and does some minor analysis on the dataset.
- prompts.json: the set of all prompts extracted from the chats.json dataset.
- t-SNE.ipynb: performs t-SNE on the prompts.json dataset to visualize the prompts in 2D space.