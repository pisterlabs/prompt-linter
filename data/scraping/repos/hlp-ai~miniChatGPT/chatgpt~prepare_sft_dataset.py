import json
from dataset import AnthropicHHRLHFDataset, DahoasRMStaticDataset


def sft_set():
    """
    A simple script to create EYLSFTStaticDataset
    """
    with open("../data/dataset_hhrlhf_train.json", "w") as fp:
        AnthropicHHRLHFDataset.save("train", fp)
    with open("../data/dataset_hhrlhf_test.json", "w") as fp:
        AnthropicHHRLHFDataset.save("test", fp)

    with open("../data/dataset_rmstatic_train.json", "w") as fp:
        DahoasRMStaticDataset.save("train", fp)
    with open("../data/dataset_rmstatic_test.json", "w") as fp:
        DahoasRMStaticDataset.save("test", fp)

    with open("../data/dataset_rmstatic_train.json") as fp:
        rmtrain = set(json.load(fp))
    with open("../data/dataset_rmstatic_test.json") as fp:
        rmtest = set(json.load(fp))

    sft_train = []
    with open("../data/dataset_hhrlhf_train.json") as fp:
        hhtrain = json.load(fp)
        for h in hhtrain:
            if h not in rmtrain:
                sft_train.append(h)

    sft_test = []
    with open("../data/dataset_hhrlhf_test.json") as fp:
        hhtest = json.load(fp)
        for h in hhtest:
            if h not in rmtest:
                sft_test.append(h)

    with open("../data/sft_train.json", "w") as fp:
        json.dump(sft_train, fp)
        print(len(sft_train))
        print(sft_train[-1])

    with open("../data/sft_test.json", "w") as fp:
        json.dump(sft_test, fp)
        print(len(sft_test))
        print(sft_test[-1])


def main():
    sft_set()


if __name__ == "__main__":
    main()
