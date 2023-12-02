import json
import pathlib, os
from tqdm import tqdm
from collections import defaultdict, Counter
from torch.utils.data import DataLoader
import openai

def run_api_inference(args, dataset):
    # format examples
    print(f"\nPreparing dataset for API...")
    examples = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, timeout=60, num_workers=1, drop_last=False)
    for bix, data in tqdm(enumerate(dataloader)):
        for i in range(len(data[0])):
            text = data[0][i]
            label = data[1][i]
            index = data[-1][i]
            incontext=defaultdict(list)
            text = dataset.get_prompt(input=text, incontext={})
            examples.append({
                "key": bix,
                "input": text,
                "gold": label
            })

    # launch api
    MY_OPENAI_KEY = args.openai_key # FILL IN YOUR API KEY from https://openai.com/api/
    assert MY_OPENAI_KEY, print(f"Please fill in your openai api key, you can obtain this from https://openai.com/api/")
    openai.api_key = MY_OPENAI_KEY
    openai.Engine.list()
    engines = openai.Engine.list()

    # run inference
    if args.model == "gpt175":
        engine_name = "text-davinci-002"
    elif args.model == "gpt6.7":
        engine_name = "text-curie-001"

    prompt_str = args.prompt_choice
    if "incontext" in args.prompt_choice:
        prompt_str += str(args.num_incontext)
    expt_path = f"{args.result_path}/{args.dataset}_{engine_name}_{args.client_subsample}clientsample_{prompt_str}_{args.num_incontext}incontext.json"

    # reload if something crashed
    if os.path.exists(expt_path):
        with open(expt_path) as f:
            examples = json.load(f)

    print(f"Running inference with {engine_name}...")
    print(f"Saving data to: {expt_path}")
    for i, example in enumerate(examples):
        if "pred" not in example:
            completion = openai.Completion.create(engine=engine_name, prompt=example['input'], temperature=0, top_p=1)
            example['pred'] = completion.choices[0].text
            
        # periodic saving in case something crashes
        if i % 100 == 0:
            print(f"Step: {i}")
            with open(expt_path, "w") as f:
                json.dump(examples, f)
            print("Saved")
            
    # save final results
    with open(expt_path, "w") as f:
        json.dump(examples, f)
        print(f"Saved API inference results to path: {expt_path}")

    # score the results
    results = []
    for ex in examples:
        if 'pred' in ex:
            results.append(ex['pred'])
    dataset.compute_accuracy(results, dataset, args)
