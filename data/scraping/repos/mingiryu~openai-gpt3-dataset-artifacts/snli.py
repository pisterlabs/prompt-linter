import openai

from loguru import logger
from datasets import load_dataset, load_metric

from fp_dataset_artifacts.utils import init_openai, get_finetune_response


INTRO = 'This is a natural language inference (NLI) classifier'


def int2label(i):
    return ['Entailment', 'Neutral', 'Contradiction'][i]


def label2int(x):
    return {'Entailment': 0, 'Neutral': 1, 'Contradiction': 2}[x]


def map_finetune(x):
    premise = x['premise']
    hypothesis = x['hypothesis']
    label = int2label(x['label'])

    return {
        'prompt': f"Premise: {premise}\n\nHypothesis: {hypothesis}\n\nLabel: ",
        'completion': label
        + '\n',  # '\n' added to prevent further text generation.
    }


def map_example(x):
    premise = x['premise']
    hypothesis = x['hypothesis']
    label = int2label(x['label'])

    return {
        'example': f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel: {label}",
    }


def map_refs_and_preds(x):
    return {
        'references': label2int(x['completion'].strip()),
        'predictions': label2int(x['response']),
    }


def get_examples(exs):
    return '\n###\n'.join(exs['example'])


def get_premises(exs):
    return '\n'.join(f'{i+1}. "{x}"' for i, x in enumerate(exs['premise']))


def get_hypotheses(exs):
    return '\n'.join(f'{i+1}. "{x}"' for i, x in enumerate(exs['hypothesis']))


def get_labels(exs):
    return '\n'.join(
        f'{i+1}: {int2label(x)}' for i, x in enumerate(exs['label'])
    )


def finetune(
    model='curie',
    n_epochs=4,
    compute_classification_metrics=True,
    classification_n_classes=3,
    train_sample_size=55000,
    train_local_filename='snli_finetune_train_sample.jsonl',
    valid_local_filename='snli_finetune_validation.jsonl',
):
    # Initialize OpenAI API with API_KEY
    init_openai()

    # Download the SNLI dataset
    data = load_dataset('snli')

    # Due to the token limitation, we can only use about 10% of the
    # training data for fine-tuning.
    train = (
        data['train'].shuffle(0).select(list(range(train_sample_size)))
    ).map(map_finetune)
    valid = data['validation'].map(map_finetune)

    # Save the data as JSON, so it can be uploaded to OpenAI.
    train.to_json(train_local_filename)
    valid.to_json(valid_local_filename)

    # Upload files
    uploaded_train_resp = openai.File.create(
        file=open(train_local_filename), purpose='fine-tune'
    )
    uploaded_valid_resp = openai.File.create(
        file=open(valid_local_filename), purpose='fine-tune'
    )

    # Get file ids
    train_remote_id = uploaded_train_resp['id']
    valid_remote_id = uploaded_valid_resp['id']

    # Create fine-tune
    finetune_resp = openai.FineTune.create(
        training_file=train_remote_id,
        validation_file=valid_remote_id,
        model=model,
        n_epochs=n_epochs,
        compute_classification_metrics=compute_classification_metrics,
        classification_n_classes=classification_n_classes,
    )

    # Get the fine-tune ID
    finetune_id = finetune_resp['id']
    logger.info(f'{finetune_id=}')

    return finetune_id


def evaluate(
    finetune_id,
    n_samples=1000,
    responses_local_filename='../results/snli_responses.jsonl',
):
    # Initialize OpenAI API with API_KEY
    init_openai()

    # Check if fine-tuning has completed.
    # And retrieve the model name.
    finetune_resp = openai.FineTune.retrieve(finetune_id)
    logger.info('Fine tuning events:')
    logger.info(finetune_resp['events'])

    assert (
        finetune_resp['events'][-1]['message'] == 'Fine-tune succeeded'
    ), 'Please wait for the fine-tuning to be completed.'

    # Get the model name
    model = finetune_resp['fine_tuned_model']
    logger.info(f'{model=}')

    # Download the SNLI dataset
    data = load_dataset('snli')

    # Evaluating on the entire test set is too costly.
    test = (
        data['test']
        .shuffle(0)
        .select(list(range(n_samples)))
        .map(map_finetune)
    )
    test = test = test.remove_columns(['premise', 'hypothesis', 'label'])

    # Evaluate the model on test set
    def map_response(x):
        try:
            response = get_finetune_response(x['prompt'], model)
            return {'response': response['choices'][0]['text']}
        except Exception as e:
            logger.warn(e)
            return {'response': None}

    responses = test.map(map_response)

    # Save the responses
    responses.to_json(responses_local_filename)

    # Load metrics
    f1_metric = load_metric('f1')
    acc_metric = load_metric('accuracy')

    # Convert response to references and predictions for metrics
    results = responses.map(map_refs_and_preds)

    # Compute metrics
    f1 = f1_metric.compute(
        references=results['references'],
        predictions=results['predictions'],
        average='weighted',
    )

    accuracy = acc_metric.compute(
        references=results['references'], predictions=results['predictions']
    )

    logger.info(f'{f1=}')
    logger.info(f'{accuracy=}')

    return model
