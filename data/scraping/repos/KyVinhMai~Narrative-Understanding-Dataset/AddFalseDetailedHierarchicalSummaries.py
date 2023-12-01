import random

import warnings

import os

import Dataloaders as dl
import settings
import openai

import numpy as np
random.seed(42)
np.random.seed(42)

from collections import Counter, namedtuple

from entity_replacement import EntityReplacer

import character_list_generator as clg
import pickle as pkl

from settings import column_separator, fake_summary_separator, line_separator

from gpt_api_processing import CreateFalseHierSummary

import time

HierarchicalSummaries = namedtuple("HierarchicalSummaries", ["summaries", "false_summaries", "summary_levels"])

def load_hierarch_summaries(hierarch_sum_folder, ent_dict_folder, num_to_process=15, start_at=0, result_folder_to_ignore=None):

    entpath = os.path.join("Data", ent_dict_folder)
    hsumpath = os.path.join("Data", hierarch_sum_folder)

    ignorepath = os.path.join("Data", result_folder_to_ignore)

    if result_folder_to_ignore is not None:
        processed_already = set([f for f in os.listdir(ignorepath) if os.path.isfile(os.path.join(ignorepath, f))])
    else:
        processed_already = set()

    sumf_to_process = [f for f in os.listdir(hsumpath) if os.path.isfile(os.path.join(hsumpath, f))]
    sumf_to_process = [s for s in sumf_to_process if s not in processed_already]
    sumf_to_process = sumf_to_process[start_at:num_to_process]

    summaries_to_process = [os.path.join(hsumpath, f) for f in sumf_to_process]
    ent_dicts_to_process = [os.path.join(entpath, f.split(".")[0] + ".repl") for f in sumf_to_process]

    assert all([os.path.isfile(f) for f in ent_dicts_to_process]), 'Missing a preprocessed replacement dictionary {}'.format(ent_dicts_to_process)

    ent_rep_dicts = []

    for p in ent_dicts_to_process:
        with open(p, "rb") as f:
            ent_rep_dicts.append(pkl.load(f))

    hsums = []

    for e in ["utf-8", "utf-16", "cp1252", "latin-1"]:
        try:
            for f in summaries_to_process:
                with open(f, "r", encoding=e) as sumfile:
                    lines = sumfile.read().split(settings.line_separator)[1:] # Ignore the header column

                    #print(len(list(zip([el.split(settings.column_separator) for el in lines if el]))))
                    #print(list(zip(*[el.split(settings.column_separator) for el in lines if el]))[0])
                    true_sums, false_sums, levels = list(zip(*[el.split(settings.column_separator) for el in lines if el]))

                    hsums.append(HierarchicalSummaries(true_sums, false_sums, levels))
            break
        except (UnicodeError, UnicodeDecodeError):
            print("Encoding issues")

    return hsums, ent_rep_dicts, sumf_to_process

def fill_false_hierarchical_summaries(hierarch_sums, hsum_files, saveprefix):

    new_hierarch_sums = []

    for i, (h, hfile) in enumerate(zip(hierarch_sums, hsum_files)):

        if i % 10 == 0:
            print("Processed {} out of {} summaries".format(i, len(hierarch_sums)))

        true_sums = h.summaries
        false_sums = h.false_summaries
        levels = h.summary_levels

        new_false_sums = []

        for t, f in zip(true_sums, false_sums):

            if f == "None":

                success = False
                for attempt in range(10):
                    try:
                        new_f = CreateFalseHierSummary(t)
                        new_false_sums.append(new_f)
                        success = True
                        break
                    except (openai.error.APIError, openai.error.Timeout, openai.error.APIConnectionError,
                            openai.error.InvalidRequestError, openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
                        print("Openai rate limit error {}. Sleeping for 5 seconds.".format(e))
                        time.sleep(5)
                    except ValueError as e:
                        print(e)
                        print("Retrying to summarize chunk")
                    except Exception as e:
                        if "That model is currently overloaded with other requests. You can retry your request, or contact us through our help center at" in str(
                                e) or "Bad gateway" in str(e):
                          print("Openai weird overloaded exception {}, {}. Sleeping for 10 seconds.".format(e, type(e).__name__))
                          time.sleep(10)
                    else:
                      raise e

                if not success:
                    new_false_sums.append("None")
            else:
                new_false_sums.append(f)

        new_hierarch_sums.append(HierarchicalSummaries(true_sums, new_false_sums, levels))


        savepath = os.path.join(saveprefix, hfile)
        with open(savepath, "w") as f:
            f.write(column_separator.join(["TRUE_SUMMARY", "FALSE_SUMMARY", "LEVEL"]))
            f.write(line_separator)

            for true_sum, false_sum, l in zip(true_sums, new_false_sums, levels):
                f.write(column_separator.join([true_sum, false_sum, str(l)]))
                f.write(line_separator)

        #
        # true_summaries.extend(new_meta_summaries)
        # false_summaries.extend(["None" for _ in range(len(new_meta_summaries))])
        # levels.extend([level for _ in range(len(new_meta_summaries))])
        #
        # current_meta_summaries = new_meta_summaries.copy()
        #
        # print("Created hierarchical summaries level {}".format(level))
        # print(current_meta_summaries[0:5])
        # print(len(current_meta_summaries))

    return new_hierarch_sums


def save_hierarch_summaries(savepath, true_summaries, false_summaries, levels):

    with open(savepath, "w") as f:
        f.write(column_separator.join(["TRUE_SUMMARY", "FALSE_SUMMARY", "LEVEL"]))
        f.write(line_separator)

        for true_sum, false_sum, l in zip(true_summaries, false_summaries, levels):
            f.write(column_separator.join([true_sum, false_sum, str(l)]))
            f.write(line_separator)


if __name__ == "__main__":

    hsums, ent_dicts, hsum_files = load_hierarch_summaries("Arseny_DETAILED_TEST_HierachialTrueOnly_1", "CharacterSubstitutionBackup", num_to_process=1000, result_folder_to_ignore="DETAILED_HIERARCH_FINAL_TEST")

    save_prefix = os.path.join("Data", "DETAILED_HIERARCH_FINAL_TEST")
    hsums_with_false = fill_false_hierarchical_summaries(hsums, hsum_files, save_prefix)


    #"MoreDetailedHierarchicalTrueSumOnlyTrainSetBackupPartial"
    #"MoreDetailedHierarchicalAddedFalseTrain"
    #
    # for hsum_wf, sumfilename in zip(hsums_with_false, hsum_files):
    #
    #     true_summaries, false_summaries, levels = hsum_wf.summaries, hsum_wf.false_summaries, hsum_wf.summary_levels
    #
    #     savepath = os.path.join("Data", "MoreDetailedHierarchicalAddedFalseTrain", sumfilename)
    #
    #     #save_hierarch_summaries(savepath, true_summaries, false_summaries, levels)

    # Save hierarchical chunks here
