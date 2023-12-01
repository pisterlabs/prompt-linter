import openai

from utils import (
    apply_heuristics,
    apply_heuristic_in_file,
    adjust_spaces,
    get_bleu_and_codebleu,
    get_env_variable,
    get_predictions_from_openai_and_write_to_file,
    read_raw_tufano_dataset_from_csv,
    write_list_to_file,
    read_dataset,
    get_EM_R4R,
    get_EM,
    format_file,
    transfer_content_to_another_file,
    get_predictions_from_edit_api_and_write_to_file,
    get_few_shot_predictions_from_openai_and_write_to_file
)

OUTPUT_DIRECTORY = "zero_shot_outputs"
TUFANO_SOURCE_FILE_PATH = "datasets/tufano/test_CC_src.txt"
TUFANO_TARGET_FILE_PATH = "datasets/tufano/test_CC_tgt.txt"
TUFANO_RAW_DATASET_FILE_PATH = "datasets/tufano/raw_test.csv"
R4R_SOURCE_FILE_PATH = "datasets/R4R/test_CC_src.txt"
R4R_TARGET_FILE_PATH = "datasets/R4R/test_CC_tgt.txt"


# get_EM(
#     f"{OUTPUT_DIRECTORY}/zero_shot_tufano_predictions_raw.txt",
#     f"{OUTPUT_DIRECTORY}/zero_shot_tufano_ground_truths_raw.txt",
#     "tufano"
# )
#
# get_bleu_and_codebleu(
#     f"{OUTPUT_DIRECTORY}/zero_shot_tufano_predictions_raw.txt",
#     f"{OUTPUT_DIRECTORY}/zero_shot_tufano_ground_truths_raw.txt"
# )
#
# apply_heuristic_in_file(f"{OUTPUT_DIRECTORY}/zero_shot_tufano_predictions_raw.txt")
#
# get_EM(
#     # f"{OUTPUT_DIRECTORY}/zero_shot_tufano_predictions_raw_applied_heuristics_also_manual.txt",
#     f"{OUTPUT_DIRECTORY}/zero_shot_tufano_predictions_raw_applied_heuristics.txt",
#     f"{OUTPUT_DIRECTORY}/zero_shot_tufano_ground_truths_raw.txt",
#     "tufano"
# )
#
#
# get_bleu_and_codebleu(
#     f"{OUTPUT_DIRECTORY}/zero_shot_tufano_predictions_raw_applied_heuristics.txt",
#     f"{OUTPUT_DIRECTORY}/zero_shot_tufano_ground_truths_raw.txt"
# )
#
# print("AFTER MANUAL")
#
# get_EM(
#     f"{OUTPUT_DIRECTORY}/zero_shot_tufano_predictions_raw_applied_heuristics_also_manual.txt",
#     f"{OUTPUT_DIRECTORY}/zero_shot_tufano_ground_truths_raw.txt",
#     "tufano"
# )
#
# get_bleu_and_codebleu(
#     f"{OUTPUT_DIRECTORY}/zero_shot_tufano_predictions_raw_applied_heuristics_also_manual.txt",
#     f"{OUTPUT_DIRECTORY}/zero_shot_tufano_ground_truths_raw.txt"
# )
#
#
# print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
# get_EM(
#     f"{OUTPUT_DIRECTORY}/zero_shot_r4r_predictions_raw.txt",
#     f"{OUTPUT_DIRECTORY}/zero_shot_r4r_ground_truths_raw.txt",
#     "R4R"
# )
#
# apply_heuristic_in_file(f"{OUTPUT_DIRECTORY}/zero_shot_r4r_predictions_raw.txt")
#
# get_EM(
#     # f"{OUTPUT_DIRECTORY}/zero_shot_r4r_predictions_raw_applied_heuristics_also_manual.txt",
#     f"{OUTPUT_DIRECTORY}/zero_shot_r4r_predictions_raw_applied_heuristics.txt",
#     f"{OUTPUT_DIRECTORY}/zero_shot_r4r_ground_truths_raw_target_only.txt",
#     "R4R"
# )

# get_bleu_and_codebleu(
#     f"{OUTPUT_DIRECTORY}/zero_shot_r4r_predictions_raw.txt",
#     f"{OUTPUT_DIRECTORY}/zero_shot_r4r_ground_truths_raw.txt"
# )
#
# get_bleu_and_codebleu(
#     f"{OUTPUT_DIRECTORY}/zero_shot_r4r_predictions_raw_applied_heuristics_also_manual.txt",
#     # f"{OUTPUT_DIRECTORY}/zero_shot_r4r_predictions_raw_applied_heuristics.txt",
#     f"{OUTPUT_DIRECTORY}/zero_shot_r4r_ground_truths_raw.txt",
# )

format_file("few_shot_outputs/few_shot_r4r_predictions_raw_no_heuristic_full.txt", adjust_spaces)