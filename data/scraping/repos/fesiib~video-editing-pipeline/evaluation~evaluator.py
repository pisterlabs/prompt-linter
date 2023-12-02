import json
import ast

import numpy as np

from backend.intent_parser import IntentParser
from backend.pipeline import Pipeline
from backend.operations import get_edit_segment
from LangChainPipeline import LangChainPipeline
from evaluation.evaluate_helpers import *
from evaluation.sentence_embedder import get_cosine_similarity_scores


# Using all metadata
intent_parser = IntentParser(50, 50)
pipeline = Pipeline(50, 0)
langchain_pipeline = LangChainPipeline(verbose=True)

# ground_truth = {
#     "editOperations": dataset[index]["edit_text"],
#     "edits": dataset[index]["temporal"],
# }

def reset_intent_parser(**props):
    intent_parser.reset(**props)

def run_gpt4(prompt_file_path, input):
    with open(prompt_file_path, 'r') as f:
        context = f.read()
    return intent_parser.completion_endpoint(context, input)

def run_pipeline_test_parser(input):
    return intent_parser.predict_relevant_text(input)

def run_pipeline_test_temporal(relevant_text):
    edits = intent_parser.predict_temporal_segments(
        relevant_text["temporal"], relevant_text["temporal_labels"])
    edits_temporal = []
    for edit in edits:
        edits_temporal.append([edit["temporalParameters"]["start"], edit["temporalParameters"]["finish"]])
    
    response = {
        "edits": edits_temporal,
        "relevant_text": relevant_text,
    }
    return response

def run_pipeline_request(edit_request):
    edit_response = intent_parser.process_request(edit_request)
    edits_temporal = []
    for edit in edit_response["edits"]:
        edits_temporal.append([edit["temporalParameters"]["start"], edit["temporalParameters"]["finish"]])
    
    response = {
        "editOperations": edit_response["requestParameters"]["editOperations"],
        "parameters": edit_response["requestParameters"]["parameters"],
        "edits": edits_temporal,
        "relevant_text": {
            "temporal": [],
            "spatial": [],
            "edit": edit_response["requestParameters"]["editOperations"],
        },
    }
    return response

def run_pipeline(input):
    relevant_text = intent_parser.predict_relevant_text(input)
    # run temporal parsing
    edits = intent_parser.predict_temporal_segments(
        relevant_text["temporal"], relevant_text["temporal_labels"])
    edits_temporal = []
    for edit in edits:
        edits_temporal.append([edit["temporalParameters"]["start"], edit["temporalParameters"]["finish"]])
    
    response = {
        "editOperations": relevant_text["edit"],
        "parameters": relevant_text["parameters"],
        "edits": edits_temporal,
        "relevant_text": relevant_text,
    }
    return response

def run_pipeline_new(input):
    relevant_text = pipeline.predict_relevant_text(input)
    edits = pipeline.predict_temporal_segments(
        relevant_text["temporal"], relevant_text["temporal_labels"], [480, 854], [
            {
                "start": 0.234,
                "finish": 60*10.2345,
            },
            {
                "start": [12, 23.3],
                "finish": [0, 15, 0.3],
            }
        ]
    )
    edits_temporal = []
    for edit in edits:
        edits_temporal.append(
            [
                edit["temporalParameters"]["start"],
                edit["temporalParameters"]["finish"],
                edit["temporalParameters"]["info"],
                edit["temporalParameters"]["source"],
            ]
        )
    
    response = {
        "editOperations": relevant_text["edit"],
        "parameters": relevant_text["parameters"],
        "edits": edits_temporal,
        "relevant_text": relevant_text,
    }
    return response

def run_pipeline_request_new(edit_request):
    edit_response = pipeline.process_request(edit_request)
    edits_temporal = []
    for edit in edit_response["edits"]:
        edits_temporal.append([
            edit["temporalParameters"]["start"],
            edit["temporalParameters"]["finish"],
            edit["temporalParameters"]["info"],
            edit["temporalParameters"]["source"],
        ])
    
    response = {
        "editOperations": edit_response["requestParameters"]["editOperations"],
        "parameters": edit_response["requestParameters"]["parameters"],
        "edits": edits_temporal,
        "relevant_text": {
            "temporal": [],
            "spatial": [],
            "edit": edit_response["requestParameters"]["editOperations"],
        },
    }
    return response

def run_langchain_pipeline_references(input):
    langchain_pipeline.set_video(input["videoId"], 10)
    references = langchain_pipeline.indexed_input_parser.run(input["text"])
    simple_references = references.get_references()
    parameters = simple_references.get_parameters_short()
    flattened_parameters = set()
    for key in parameters.keys():
        flattened_parameters = flattened_parameters.union(parameters[key])
    flattened_parameters = list(flattened_parameters)

    response = {
        "editOperations": simple_references.edit,
        "temporal": simple_references.temporal,
        "spatial": simple_references.spatial,
        "edit": [item.reference for item in references.edit_references],
        "parameters": flattened_parameters,
    }
    return response

def run_langchain_pipeline_temporal_indexed(input):
    return run_langchain_pipeline_temporal(input, indexed=True)

def run_langchain_pipeline_temporal(input, indexed=False):
    langchain_pipeline.set_video(input["videoId"], 10)
    references = None
    temporal = []
    temporal_labels = []
    temporal_offsets = []
    if indexed == True:
        references = langchain_pipeline.indexed_input_parser.run(input["text"])
        temporal = [item.reference for item in references.temporal_references]
        temporal_labels = references.temporal_labels
        temporal_offsets = [item.offset for item in references.temporal_references]
    else:
        references = langchain_pipeline.input_parser.run(input["text"])
        temporal = references.temporal
        temporal_labels = references.temporal_labels
        temporal_offsets = [-1 for _ in temporal]

    edits = langchain_pipeline.predict_temporal_segments(
        input["text"],
        temporal, temporal_labels, temporal_offsets,
        0, input["sketch_timestamp"],
        input["video_shape"], [],
        input["video_duration"],
    )
    edits_temporal = []
    edits_temporal_reasoning = []
    edits_spatial = []
    edits_spatial_reasoning = []
    for edit in edits:
        edits_temporal.append([
            edit["temporalParameters"]["start"],
            edit["temporalParameters"]["finish"],
        ])
        edits_temporal_reasoning.append([
            edit["temporalParameters"]["info"],
            edit["temporalParameters"]["source"],
            edit["temporalParameters"]["offsets"],
        ])
        edits_spatial.append(edit["spatialParameters"])
        edits_spatial_reasoning.append([
            edit["spatialParameters"]["info"],
            edit["spatialParameters"]["source"],
            edit["temporalParameters"]["offsets"],
        ])
    
    response = {
        "editOperations": references.edit,
        "parameters": references.get_parameters_short(),
        "edits": edits_temporal,
        "edits_temporal_reasoning": edits_temporal_reasoning,
        "edits_spatial": edits_spatial,
        "edits_spatial_reasoning": edits_spatial_reasoning,
        "relevant_text": {
            "temporal": temporal,
            "spatial": [item.reference for item in references.spatial_references],
            "edit": [item.reference for item in references.edit_references],
            "indexed_temporal": [[item.offset, item.reference] for item in references.temporal_references],
            "indexed_spatial": [[item.offset, item.reference] for item in references.spatial_references],
            "indexed_edit": [[item.offset, item.reference] for item in references.edit_references],
        },
    }
    return response

def run_langchain_pipeline_request(edit_request):
    langchain_pipeline.set_video(edit_request["videoId"], 10)
    edit_response = langchain_pipeline.process_request_indexed(edit_request)
    edits_temporal = []
    edits_temporal_reasoning = []
    edits_spatial = []
    edits_spatial_reasoning = []
    for edit in edit_response["edits"]:
        edits_temporal.append([
            edit["temporalParameters"]["start"],
            edit["temporalParameters"]["finish"],
        ])
        edits_temporal_reasoning.append([
            edit["temporalParameters"]["info"],
            edit["temporalParameters"]["source"],
            edit["temporalParameters"]["offsets"],
        ])
        edits_spatial.append(edit["spatialParameters"])
        print(edit["spatialParameters"])
        edits_spatial_reasoning.append([
            edit["spatialParameters"]["info"],
            edit["spatialParameters"]["source"],
            edit["temporalParameters"]["offsets"],
        ])
    
    relevant_text = edit_response["requestParameters"]["relevantText"]

    response = {
        "editOperations": edit_response["requestParameters"]["editOperations"],
        "parameters": edit_response["requestParameters"]["parameters"],
        "edits": edits_temporal,
        "edits_temporal_reasoning": edits_temporal_reasoning,
        "edits_spatial": edits_spatial,
        "edits_spatial_reasoning": edits_spatial_reasoning,
        "relevant_text": {
            "temporal": relevant_text["temporal"],
            "spatial": relevant_text["spatial"],
            "edit": relevant_text["edit"],
        },
    }
    return response

def run_evaluation_for_task(
    task_id = 6,
    data_point_getter = get_data_point_as_request,
    pipeline_runner = run_langchain_pipeline_request,
    indexes = []
):
    dataset = get_dataset_for_task(task_id)

    average_temporal_f1_0 = 0
    average_temporal_precision_0 = 0
    average_temporal_recall_0 = 0
    average_temporal_f1_5 = 0
    average_temporal_precision_5 = 0
    average_temporal_recall_5 = 0
    average_temporal_f1_10 = 0
    average_temporal_precision_10 = 0
    average_temporal_recall_10 = 0

    average_spatial_miou_0 = 0
    average_spatial_thresholded_0 = 0
    average_spatial_miou_5 = 0
    average_spatial_thresholded_5 = 0
    average_spatial_miou_10 = 0
    average_spatial_thresholded_10 = 0

    average_edit_operation_f1 = 0
    average_edit_operation_precision = 0
    average_edit_operation_recall = 0

    all_temporal_f1_0 = []
    all_temporal_precision_0 = []
    all_temporal_recall_0 = []
    all_temporal_f1_5 = []
    all_temporal_precision_5 = []
    all_temporal_recall_5 = []
    all_temporal_f1_10 = []
    all_temporal_precision_10 = []
    all_temporal_recall_10 = []
    
    all_spatial_miou_0 = []
    all_spatial_thresholded_0 = []
    all_spatial_miou_5 = []
    all_spatial_thresholded_5 = []
    all_spatial_miou_10 = []
    all_spatial_thresholded_10 = []

    all_edit_operation = []

    all_cosine_similarity_temporal = []
    all_top_10_cosine_similarity_temporal = []
    
    all_cosine_similarity_spatial = []
    all_top_10_cosine_similarity_spatial = []

    if (len(dataset) == 0):
        return {
            "temporal_f1_0": 1,
            "temporal_precision_0": 1,
            "temporal_recall_0": 1,
            "temporal_f1_5": 1,
            "temporal_precision_5": 1,
            "temporal_recall_5": 1,
            "temporal_f1_10": 1,
            "temporal_precision_10": 1,
            "temporal_recall_10": 1,
            "spatial_miou_0": 1,
            "spatial_thresholded_0": 1,
            "spatial_miou_5": 1,
            "spatial_thresholded_5": 1,
            "spatial_miou_10": 1,
            "spatial_thresholded_10": 1,

            "edit_operation_f1": 1,
            "edit_operation_precision": 1,
            "edit_operation_recall": 1,

            "all_temporal_f1_0": all_temporal_f1_0,
            "all_temporal_precision_0": all_temporal_precision_0,
            "all_temporal_recall_0": all_temporal_recall_0,
            "all_temporal_f1_5": all_temporal_f1_5,
            "all_temporal_precision_5": all_temporal_precision_5,
            "all_temporal_recall_5": all_temporal_recall_5,
            "all_temporal_f1_10": all_temporal_f1_10,
            "all_temporal_precision_10": all_temporal_precision_10,
            "all_temporal_recall_10": all_temporal_recall_10,
            "all_spatial_miou_0": all_spatial_miou_0,
            "all_spatial_thresholded_0": all_spatial_thresholded_0,
            "all_spatial_miou_5": all_spatial_miou_5,
            "all_spatial_thresholded_5": all_spatial_thresholded_5,
            "all_spatial_miou_10": all_spatial_miou_10,
            "all_spatial_thresholded_10": all_spatial_thresholded_10,
            "all_edit_operation": all_edit_operation,
            "all_cosine_similarity_temporal": all_cosine_similarity_temporal,
            "all_top_10_cosine_similarity_temporal": all_top_10_cosine_similarity_temporal,
            "all_cosine_similarity_spatial": all_cosine_similarity_spatial,
            "all_top_10_cosine_similarity_spatial": all_top_10_cosine_similarity_spatial,
            "dataset": dataset,
        }
    
    if (len(indexes) == 0):
        indexes = range(len(dataset))

    indexes = [i for i in indexes if i < len(dataset)]

    for index in indexes:
        data_point = data_point_getter(dataset, index)
        input = data_point[0]
        ground_truth = data_point[1]
        prediction = pipeline_runner(input)

        # cosine similarity if possible
        cosine_scores_temporal, top_10_pairs_temporal = get_cosine_similarity_scores(
            prediction["relevant_text"]["temporal"],
            ground_truth["relevant_text"]["temporal"]
        )
        cosine_scores_spatial, top_10_pairs_spatial = get_cosine_similarity_scores(
            prediction["relevant_text"]["spatial"],
            ground_truth["relevant_text"]["spatial"]
        )

        all_top_10_cosine_similarity_temporal.append(top_10_pairs_temporal)
        all_cosine_similarity_temporal.append(cosine_scores_temporal)

        (
            ( f1_0, precision_0, recall_0 ),
            ( f1_5, precision_5, recall_5 ),
            ( f1_10, precision_10, recall_10),
        ) = get_temporal_evaluation(prediction["edits"], ground_truth["edits"])
        
        (
            (miou_0, thresholded_0),
            (miou_5, thresholded_5),
            (miou_10, thresholded_10),
        ) = get_spatial_evaluation(
            prediction["edits_spatial"],
            ground_truth["edits_spatial"],
            prediction["edits"],
            ground_truth["edits"],
            iou_threshold=0.5
        )
        
        edit_operation = get_edit_operation_evaluation(prediction["editOperations"], ground_truth["editOperations"])

        average_temporal_f1_0 += f1_0
        average_temporal_precision_0 += precision_0
        average_temporal_recall_0 += recall_0
        average_temporal_f1_5 += f1_5
        average_temporal_precision_5 += precision_5
        average_temporal_recall_5 += recall_5
        average_temporal_f1_10 += f1_10
        average_temporal_precision_10 += precision_10
        average_temporal_recall_10 += recall_10

        average_spatial_miou_0 += miou_0
        average_spatial_thresholded_0 += thresholded_0
        average_spatial_miou_5 += miou_5
        average_spatial_thresholded_5 += thresholded_5
        average_spatial_miou_10 += miou_10
        average_spatial_thresholded_10 += thresholded_10

        average_edit_operation_f1 += edit_operation[0]
        average_edit_operation_precision += edit_operation[1]
        average_edit_operation_recall += edit_operation[2]

        all_temporal_f1_0.append(f1_0)
        all_temporal_precision_0.append(precision_0)
        all_temporal_recall_0.append(recall_0)
        all_temporal_f1_5.append(f1_5)
        all_temporal_precision_5.append(precision_5)
        all_temporal_recall_5.append(recall_5)
        all_temporal_f1_10.append(f1_10)
        all_temporal_precision_10.append(precision_10)
        all_temporal_recall_10.append(recall_10)

        all_spatial_miou_0.append(miou_0)
        all_spatial_thresholded_0.append(thresholded_0)
        all_spatial_miou_5.append(miou_5)
        all_spatial_thresholded_5.append(thresholded_5)
        all_spatial_miou_10.append(miou_10)
        all_spatial_thresholded_10.append(thresholded_10)

        all_edit_operation.append(edit_operation)

        print("--------------------")
        print("!!!input!!!: ", input)
        print("!!!prediction!!!: ", prediction)
        print("!!!ground_truth!!!: ", ground_truth)
        print("!!!temporal evaluation margin=0!!!: ", "f1-margin-0: ", f1_0, "precision-margin-0: ", precision_0, "recall-margin-0: ", recall_0)
        print("!!!temporal evaluation margin=5!!!: ", "f1-margin-5: ", f1_5, "precision-margin-5: ", precision_5, "recall-margin-5: ", recall_5)
        print("!!!temporal evaluation margin=10!!!: ", "f1-margin-10: ", f1_10, "precision-margin-10: ", precision_10, "recall-margin-10: ", recall_10)
        print("!!!spatial evaluation margin=0!!!: ", "miou-margin-0: ", miou_0, "thresholded-margin-0: ", thresholded_0)
        print("!!!spatial evaluation margin=5!!!: ", "miou-margin-5: ", miou_5, "thresholded-margin-5: ", thresholded_5)
        print("!!!spatial evaluation margin=10!!!: ", "miou-margin-10: ", miou_10, "thresholded-margin-10: ", thresholded_10)
        print("!!!edit_op evaluation!!!: ", "f1", edit_operation[0], "precision", edit_operation[1], "recall", edit_operation[2])
        print("--------------------")
        print("!!!(temporal)cosine_similarity!!!: ", cosine_scores_temporal)
        print("!!!(temporal)top_4_cosine_similarity!!!: ", json.dumps(top_10_pairs_temporal[0:4], indent=1))
        print("--------------------")
        print("!!!(spatial)cosine_similarity!!!: ", cosine_scores_spatial)
        print("!!!(spatial)top_4_cosine_similarity!!!: ", json.dumps(top_10_pairs_spatial[0:4], indent=1))
        print("--------------------")

    average_temporal_f1_0 /= len(indexes)
    average_temporal_precision_0 /= len(indexes)
    average_temporal_recall_0 /= len(indexes)
    average_temporal_f1_5 /= len(indexes)
    average_temporal_precision_5 /= len(indexes)
    average_temporal_recall_5 /= len(indexes)
    average_temporal_f1_10 /= len(indexes)
    average_temporal_precision_10 /= len(indexes)
    average_temporal_recall_10 /= len(indexes)
    average_spatial_miou_0 /= len(indexes)
    average_spatial_thresholded_0 /= len(indexes)
    average_spatial_miou_5 /= len(indexes)
    average_spatial_thresholded_5 /= len(indexes)
    average_spatial_miou_10 /= len(indexes)
    average_spatial_thresholded_10 /= len(indexes)
    average_edit_operation_f1 /= len(indexes)
    average_edit_operation_precision /= len(indexes)
    average_edit_operation_recall /= len(indexes)
    return {
        "temporal_f1_0": average_temporal_f1_0,
        "temporal_precision_0": average_temporal_precision_0,
        "temporal_recall_0": average_temporal_recall_0,
        "temporal_f1_5": average_temporal_f1_5,
        "temporal_precision_5": average_temporal_precision_5,
        "temporal_recall_5": average_temporal_recall_5,
        "temporal_f1_10": average_temporal_f1_10,
        "temporal_precision_10": average_temporal_precision_10,
        "temporal_recall_10": average_temporal_recall_10,
        "spatial_miou_0": average_spatial_miou_0,
        "spatial_thresholded_0": average_spatial_thresholded_0,
        "spatial_miou_5": average_spatial_miou_5,
        "spatial_thresholded_5": average_spatial_thresholded_5,
        "spatial_miou_10": average_spatial_miou_10,
        "spatial_thresholded_10": average_spatial_thresholded_10,
        "edit_operation_f1": average_edit_operation_f1,
        "edit_operation_precision": average_edit_operation_precision,
        "edit_operation_recall": average_edit_operation_recall,
        "all_temporal_f1_0": all_temporal_f1_0,
        "all_temporal_precision_0": all_temporal_precision_0,
        "all_temporal_recall_0": all_temporal_recall_0,
        "all_temporal_f1_5": all_temporal_f1_5,
        "all_temporal_precision_5": all_temporal_precision_5,
        "all_temporal_recall_5": all_temporal_recall_5,
        "all_temporal_f1_10": all_temporal_f1_10,
        "all_temporal_precision_10": all_temporal_precision_10,
        "all_temporal_recall_10": all_temporal_recall_10,
        "all_spatial_miou_0": all_spatial_miou_0,
        "all_spatial_thresholded_0": all_spatial_thresholded_0,
        "all_spatial_miou_5": all_spatial_miou_5,
        "all_spatial_thresholded_5": all_spatial_thresholded_5,
        "all_spatial_miou_10": all_spatial_miou_10,
        "all_spatial_thresholded_10": all_spatial_thresholded_10,
        "all_edit_operation": all_edit_operation,
        "all_cosine_similarity_temporal": all_cosine_similarity_temporal,
        "all_top_10_cosine_similarity_temporal": all_top_10_cosine_similarity_temporal,
        "all_cosine_similarity_spatial": all_cosine_similarity_spatial,
        "all_top_10_cosine_similarity_spatial": all_top_10_cosine_similarity_spatial,    
        "dataset": [item for i, item in enumerate(dataset) if i in indexes],
    }


def run_evaluation( 
    task_ids,
    data_point_getter = get_data_point_as_request,
    pipeline_runner = run_langchain_pipeline_request,
    indexes = []
):
    average_temporal_f1_0 = 0
    average_temporal_precision_0 = 0
    average_temporal_recall_0 = 0
    average_temporal_f1_5 = 0
    average_temporal_precision_5 = 0
    average_temporal_recall_5 = 0
    average_temporal_f1_10 = 0
    average_temporal_precision_10 = 0
    average_temporal_recall_10 = 0

    average_spatial_miou_0 = 0
    average_spatial_thresholded_0 = 0
    average_spatial_miou_5 = 0
    average_spatial_thresholded_5 = 0
    average_spatial_miou_10 = 0
    average_spatial_thresholded_10 = 0

    average_edit_operation_f1 = 0
    average_edit_operation_precision = 0
    average_edit_operation_recall = 0

    all_temporal_f1_0 = []
    all_temporal_precision_0 = []
    all_temporal_recall_0 = []
    all_temporal_f1_5 = []
    all_temporal_precision_5 = []
    all_temporal_recall_5 = []
    all_temporal_f1_10 = []
    all_temporal_precision_10 = []
    all_temporal_recall_10 = []
    
    all_spatial_miou_0 = []
    all_spatial_thresholded_0 = []
    all_spatial_miou_5 = []
    all_spatial_thresholded_5 = []
    all_spatial_miou_10 = []
    all_spatial_thresholded_10 = []

    all_edit_operation = []

    all_cosine_similarity_temporal = []
    all_top_10_cosine_similarity_temporal = []
    
    all_cosine_similarity_spatial = []
    all_top_10_cosine_similarity_spatial = []

    evaluated_dataset = []
    for task_id in task_ids:
        dataset = get_dataset_for_task(task_id)
        if (len(dataset) == 0):
            continue
        cur_indexes = []
        if (len(indexes) == 0):
            cur_indexes = range(len(dataset))
        else:
            for index in indexes:
                if index < len(dataset):
                    cur_indexes.append(index)

        if (len(cur_indexes) == 0):
            continue

        task_temporal_f1_0 = []
        task_temporal_precision_0 = []
        task_temporal_recall_0 = []
        task_temporal_f1_5 = []
        task_temporal_precision_5 = []
        task_temporal_recall_5 = []
        task_temporal_f1_10 = []
        task_temporal_precision_10 = []
        task_temporal_recall_10 = []

        task_spatial_miou_0 = []
        task_spatial_thresholded_0 = []
        task_spatial_miou_5 = []
        task_spatial_thresholded_5 = []
        task_spatial_miou_10 = []
        task_spatial_thresholded_10 = []

        task_edit_operation_f1 = []
        task_edit_operation_precision = []
        task_edit_operation_recall = []

        for index in cur_indexes:
            evaluated_dataset.append(dataset[index])
            data_point = data_point_getter(dataset, index)
            input = data_point[0]
            ground_truth = data_point[1]
            prediction = pipeline_runner(input)

            # cosine similarity if possible
            cosine_scores_temporal, top_10_pairs_temporal = get_cosine_similarity_scores(
                prediction["relevant_text"]["temporal"],
                ground_truth["relevant_text"]["temporal"]
            )
            cosine_scores_spatial, top_10_pairs_spatial = get_cosine_similarity_scores(
                prediction["relevant_text"]["spatial"],
                ground_truth["relevant_text"]["spatial"]
            )

            all_top_10_cosine_similarity_temporal.append(top_10_pairs_temporal)
            all_cosine_similarity_temporal.append(cosine_scores_temporal)
            (
                ( f1_0, precision_0, recall_0 ),
                ( f1_5, precision_5, recall_5 ),
                ( f1_10, precision_10, recall_10),
            ) = get_temporal_evaluation(prediction["edits"], ground_truth["edits"])
            
            (
                (miou_0, thresholded_0),
                (miou_5, thresholded_5),
                (miou_10, thresholded_10),
            ) = get_spatial_evaluation(
                prediction["edits_spatial"],
                ground_truth["edits_spatial"],
                prediction["edits"],
                ground_truth["edits"],
                iou_threshold=0.5
            )
            
            edit_operation = get_edit_operation_evaluation(prediction["editOperations"], ground_truth["editOperations"])

            average_temporal_f1_0 += f1_0
            average_temporal_precision_0 += precision_0
            average_temporal_recall_0 += recall_0
            average_temporal_f1_5 += f1_5
            average_temporal_precision_5 += precision_5
            average_temporal_recall_5 += recall_5
            average_temporal_f1_10 += f1_10
            average_temporal_precision_10 += precision_10
            average_temporal_recall_10 += recall_10

            average_spatial_miou_0 += miou_0
            average_spatial_thresholded_0 += thresholded_0
            average_spatial_miou_5 += miou_5
            average_spatial_thresholded_5 += thresholded_5
            average_spatial_miou_10 += miou_10
            average_spatial_thresholded_10 += thresholded_10

            average_edit_operation_f1 += edit_operation[0]
            average_edit_operation_precision += edit_operation[1]
            average_edit_operation_recall += edit_operation[2]

            all_temporal_f1_0.append(f1_0)
            all_temporal_precision_0.append(precision_0)
            all_temporal_recall_0.append(recall_0)
            all_temporal_f1_5.append(f1_5)
            all_temporal_precision_5.append(precision_5)
            all_temporal_recall_5.append(recall_5)
            all_temporal_f1_10.append(f1_10)
            all_temporal_precision_10.append(precision_10)
            all_temporal_recall_10.append(recall_10)

            all_spatial_miou_0.append(miou_0)
            all_spatial_thresholded_0.append(thresholded_0)
            all_spatial_miou_5.append(miou_5)
            all_spatial_thresholded_5.append(thresholded_5)
            all_spatial_miou_10.append(miou_10)
            all_spatial_thresholded_10.append(thresholded_10)

            all_edit_operation.append(edit_operation)

            task_temporal_f1_0.append(f1_0)
            task_temporal_precision_0.append(precision_0)
            task_temporal_recall_0.append(recall_0)
            task_temporal_f1_5.append(f1_5)
            task_temporal_precision_5.append(precision_5)
            task_temporal_recall_5.append(recall_5)
            task_temporal_f1_10.append(f1_10)
            task_temporal_precision_10.append(precision_10)
            task_temporal_recall_10.append(recall_10)
            
            task_spatial_miou_0.append(miou_0)
            task_spatial_thresholded_0.append(thresholded_0)
            task_spatial_miou_5.append(miou_5)
            task_spatial_thresholded_5.append(thresholded_5)
            task_spatial_miou_10.append(miou_10)
            task_spatial_thresholded_10.append(thresholded_10)

            task_edit_operation_f1.append(edit_operation[0])
            task_edit_operation_precision.append(edit_operation[1])
            task_edit_operation_recall.append(edit_operation[2])

            print("--------------------")
            print("!!!input!!!: ", input)
            print("!!!prediction!!!: ", prediction)
            print("!!!ground_truth!!!: ", ground_truth)
            print("!!!temporal evaluation margin=0!!!: ", "f1-margin-0: ", f1_0, "precision-margin-0: ", precision_0, "recall-margin-0: ", recall_0)
            print("!!!temporal evaluation margin=5!!!: ", "f1-margin-5: ", f1_5, "precision-margin-5: ", precision_5, "recall-margin-5: ", recall_5)
            print("!!!temporal evaluation margin=10!!!: ", "f1-margin-10: ", f1_10, "precision-margin-10: ", precision_10, "recall-margin-10: ", recall_10)
            print("!!!spatial evaluation margin=0!!!: ", "miou-margin-0: ", miou_0, "thresholded-margin-0: ", thresholded_0)
            print("!!!spatial evaluation margin=5!!!: ", "miou-margin-5: ", miou_5, "thresholded-margin-5: ", thresholded_5)
            print("!!!spatial evaluation margin=10!!!: ", "miou-margin-10: ", miou_10, "thresholded-margin-10: ", thresholded_10)
            print("!!!edit_op evaluation!!!: ", "f1", edit_operation[0], "precision", edit_operation[1], "recall", edit_operation[2])
            print("--------------------")
            print("!!!(temporal)cosine_similarity!!!: ", cosine_scores_temporal)
            print("!!!(temporal)top_4_cosine_similarity!!!: ", json.dumps(top_10_pairs_temporal[0:4], indent=1))
            print("--------------------")
            print("!!!(spatial)cosine_similarity!!!: ", cosine_scores_spatial)
            print("!!!(spatial)top_4_cosine_similarity!!!: ", json.dumps(top_10_pairs_spatial[0:4], indent=1))
            print("--------------------")

        print("Statistics for task: ", task_id)
        print("--------------------")
        print("!!!temporal evaluation margin=0!!!: ", "f1-margin-0: ", np.mean(task_temporal_f1_0), "precision-margin-0: ", np.mean(task_temporal_precision_0), "recall-margin-0: ", np.mean(task_temporal_recall_0))
        print("!!!temporal evaluation margin=5!!!: ", "f1-margin-5: ", np.mean(task_temporal_f1_5), "precision-margin-5: ", np.mean(task_temporal_precision_5), "recall-margin-5: ", np.mean(task_temporal_recall_5))
        print("!!!temporal evaluation margin=10!!!: ", "f1-margin-10: ", np.mean(task_temporal_f1_10), "precision-margin-10: ", np.mean(task_temporal_precision_10), "recall-margin-10: ", np.mean(task_temporal_recall_10))
        print("!!!spatial evaluation margin=0!!!: ", "miou-margin-0: ", np.mean(task_spatial_miou_0), "thresholded-margin-0: ", np.mean(task_spatial_thresholded_0))
        print("!!!spatial evaluation margin=5!!!: ", "miou-margin-5: ", np.mean(task_spatial_miou_5), "thresholded-margin-5: ", np.mean(task_spatial_thresholded_5))
        print("!!!spatial evaluation margin=10!!!: ", "miou-margin-10: ", np.mean(task_spatial_miou_10), "thresholded-margin-10: ", np.mean(task_spatial_thresholded_10))
        print("!!!edit_op evaluation!!!: ", "f1", np.mean(task_edit_operation_f1), "precision", np.mean(task_edit_operation_precision), "recall", np.mean(task_edit_operation_recall))
        print("--------------------")


    if len(evaluated_dataset) == 0:
        average_temporal_f1_0 = 1
        average_temporal_precision_0 = 1
        average_temporal_recall_0 = 1
        average_temporal_f1_5 = 1
        average_temporal_precision_5 = 1
        average_temporal_recall_5 = 1
        average_temporal_f1_10 = 1
        average_temporal_precision_10 = 1
        average_temporal_recall_10 = 1
        average_spatial_miou_0 = 1
        average_spatial_thresholded_0 = 1
        average_spatial_miou_5 = 1
        average_spatial_thresholded_5 = 1
        average_spatial_miou_10 = 1
        average_spatial_thresholded_10 = 1
        average_edit_operation_f1 = 1
        average_edit_operation_precision = 1
        average_edit_operation_recall = 1
    else:
        average_temporal_f1_0 /= len(evaluated_dataset)
        average_temporal_precision_0 /= len(evaluated_dataset)
        average_temporal_recall_0 /= len(evaluated_dataset)
        average_temporal_f1_5 /= len(evaluated_dataset)
        average_temporal_precision_5 /= len(evaluated_dataset)
        average_temporal_recall_5 /= len(evaluated_dataset)
        average_temporal_f1_10 /= len(evaluated_dataset)
        average_temporal_precision_10 /= len(evaluated_dataset)
        average_temporal_recall_10 /= len(evaluated_dataset)
        average_spatial_miou_0 /= len(evaluated_dataset)
        average_spatial_thresholded_0 /= len(evaluated_dataset)
        average_spatial_miou_5 /= len(evaluated_dataset)
        average_spatial_thresholded_5 /= len(evaluated_dataset)
        average_spatial_miou_10 /= len(evaluated_dataset)
        average_spatial_thresholded_10 /= len(evaluated_dataset)
        average_edit_operation_f1 /= len(evaluated_dataset)
        average_edit_operation_precision /= len(evaluated_dataset)
        average_edit_operation_recall /= len(evaluated_dataset)

    return {
        "temporal_f1_0": average_temporal_f1_0,
        "temporal_precision_0": average_temporal_precision_0,
        "temporal_recall_0": average_temporal_recall_0,
        "temporal_f1_5": average_temporal_f1_5,
        "temporal_precision_5": average_temporal_precision_5,
        "temporal_recall_5": average_temporal_recall_5,
        "temporal_f1_10": average_temporal_f1_10,
        "temporal_precision_10": average_temporal_precision_10,
        "temporal_recall_10": average_temporal_recall_10,
        "spatial_miou_0": average_spatial_miou_0,
        "spatial_thresholded_0": average_spatial_thresholded_0,
        "spatial_miou_5": average_spatial_miou_5,
        "spatial_thresholded_5": average_spatial_thresholded_5,
        "spatial_miou_10": average_spatial_miou_10,
        "spatial_thresholded_10": average_spatial_thresholded_10,
        "edit_operation_f1": average_edit_operation_f1,
        "edit_operation_precision": average_edit_operation_precision,
        "edit_operation_recall": average_edit_operation_recall,
        "all_temporal_f1_0": all_temporal_f1_0,
        "all_temporal_precision_0": all_temporal_precision_0,
        "all_temporal_recall_0": all_temporal_recall_0,
        "all_temporal_f1_5": all_temporal_f1_5,
        "all_temporal_precision_5": all_temporal_precision_5,
        "all_temporal_recall_5": all_temporal_recall_5,
        "all_temporal_f1_10": all_temporal_f1_10,
        "all_temporal_precision_10": all_temporal_precision_10,
        "all_temporal_recall_10": all_temporal_recall_10,
        "all_spatial_miou_0": all_spatial_miou_0,
        "all_spatial_thresholded_0": all_spatial_thresholded_0,
        "all_spatial_miou_5": all_spatial_miou_5,
        "all_spatial_thresholded_5": all_spatial_thresholded_5,
        "all_spatial_miou_10": all_spatial_miou_10,
        "all_spatial_thresholded_10": all_spatial_thresholded_10,
        "all_edit_operation": all_edit_operation,
        "all_cosine_similarity_temporal": all_cosine_similarity_temporal,
        "all_top_10_cosine_similarity_temporal": all_top_10_cosine_similarity_temporal,
        "all_cosine_similarity_spatial": all_cosine_similarity_spatial,
        "all_top_10_cosine_similarity_spatial": all_top_10_cosine_similarity_spatial,    
        "dataset": evaluated_dataset,
    }

def run_evaluation_spatial(
    task_ids,
    indexes = []
):
    ### spatial evaluation in isolation (given ground_truth segments)
    all_spatial_miou_0 = []
    all_spatial_thresholded_0 = []
    all_spatial_miou_5 = []
    all_spatial_thresholded_5 = []
    all_spatial_miou_10 = []
    all_spatial_thresholded_10 = []

    all_spatial_miou = []
    all_spatial_thresholded = []

    evaluated_dataset = []

    for task_id in task_ids:
        dataset = get_dataset_for_task(task_id)
        if (len(dataset) == 0):
            continue
        cur_indexes = []
        if (len(indexes) == 0):
            cur_indexes = range(len(dataset))
        else:
            for index in indexes:
                if index < len(dataset):
                    cur_indexes.append(index)

        if (len(cur_indexes) == 0):
            continue

        task_spatial_miou_0 = []
        task_spatial_thresholded_0 = []
        task_spatial_miou_5 = []
        task_spatial_thresholded_5 = []
        task_spatial_miou_10 = []
        task_spatial_thresholded_10 = []
        task_spatial_miou = []
        task_spatial_thresholded = []

        for index in cur_indexes:
            data_point = get_data_point(dataset, index)
            input = data_point[0]
            ground_truth = data_point[1]
            count_spatial_gt = 0
            for spatial_gts in ground_truth["edits_spatial"]:
                if len(spatial_gts) > 0:
                    count_spatial_gt += 1

            if count_spatial_gt == 0:
                continue
            
            evaluated_dataset.append(dataset[index])
            langchain_pipeline.set_video(input["videoId"], 10)

            sketches = input["sketch"]
            sketch_timestamp = input["sketch_timestamp"]
            for sketch in sketches:
                sketch["timestamp"] = sketch_timestamp
            video_shape = input["video_shape"]

            edits = []
            for edit in ground_truth["edits"]:
                start = edit[0]
                finish = edit[1]
                explanation = ["ground_truth"]
                source = ["ground_truth"]
                offsets = [-1]
                edit = get_edit_segment(start, finish, explanation, source, offsets, video_shape)
                edits.append(edit)       

            references = langchain_pipeline.indexed_input_parser.run(input["text"])
            simple_references = references.get_references()
            edits =  langchain_pipeline.predict_spatial_locations_new(
                input["text"],
                simple_references.spatial, simple_references.spatial_labels,
                [item.offset for item in references.spatial_references],
                edits, sketches, video_shape,
                sketch_timestamp
            )
            edits_temporal = []
            edits_temporal_reasoning = []
            edits_spatial = []
            edits_spatial_reasoning = []
            for edit in edits:
                edits_temporal.append([
                    edit["temporalParameters"]["start"],
                    edit["temporalParameters"]["finish"],
                ])
                edits_temporal_reasoning.append([
                    edit["temporalParameters"]["info"],
                    edit["temporalParameters"]["source"],
                    edit["temporalParameters"]["offsets"],
                ])
                edits_spatial.append(edit["spatialParameters"])
                edits_spatial_reasoning.append([
                    edit["spatialParameters"]["info"],
                    edit["spatialParameters"]["source"],
                    edit["temporalParameters"]["offsets"],
                ])
            prediction = {
                "editOperations": simple_references.edit,
                "parameters": simple_references.get_parameters_short(),
                "edits": edits_temporal,
                "edits_temporal_reasoning": edits_temporal_reasoning,
                "edits_spatial": edits_spatial,
                "edits_spatial_reasoning": edits_spatial_reasoning,
                "relevant_text": {
                    "temporal": simple_references.temporal,
                    "spatial": simple_references.spatial,
                    "edit": [item.reference for item in references.edit_references],
                    "parameters": simple_references.get_parameters(),
                },
            }

            (
                (miou_0, thresholded_0),
                (miou_5, thresholded_5),
                (miou_10, thresholded_10),
            ) = get_spatial_evaluation(
                prediction["edits_spatial"],
                ground_truth["edits_spatial"],
                prediction["edits"],
                ground_truth["edits"],
                iou_threshold=0.5
            )

            (miou, thresholded) = get_spatial_evaluation_pairs(
                prediction["edits_spatial"],
                ground_truth["edits_spatial"],
                iou_threshold=0.5
            )

            all_spatial_miou_0.append(miou_0)
            all_spatial_thresholded_0.append(thresholded_0)
            all_spatial_miou_5.append(miou_5)
            all_spatial_thresholded_5.append(thresholded_5)
            all_spatial_miou_10.append(miou_10)
            all_spatial_thresholded_10.append(thresholded_10)

            all_spatial_miou.append(miou)
            all_spatial_thresholded.append(thresholded)

            task_spatial_miou_0.append(miou_0)
            task_spatial_thresholded_0.append(thresholded_0)
            task_spatial_miou_5.append(miou_5)
            task_spatial_thresholded_5.append(thresholded_5)
            task_spatial_miou_10.append(miou_10)
            task_spatial_thresholded_10.append(thresholded_10)

            task_spatial_miou.append(miou)
            task_spatial_thresholded.append(thresholded)

            print("--------------------")
            print("!!!input!!!: ", input)
            print("!!!compared count!!!: ", count_spatial_gt)
            print("!!!prediction!!!: ", prediction)
            print("!!!ground_truth!!!: ", ground_truth)
            print("!!!spatial evaluation margin=0!!!: ", "miou-margin-0: ", miou_0, "thresholded-margin-0: ", thresholded_0)
            print("!!!spatial evaluation margin=5!!!: ", "miou-margin-5: ", miou_5, "thresholded-margin-5: ", thresholded_5)
            print("!!!spatial evaluation margin=10!!!: ", "miou-margin-10: ", miou_10, "thresholded-margin-10: ", thresholded_10)
            print("!!!spatial evaluiation pairs!!!: ", "miou: ", miou, "thresholded: ", thresholded)
            print("--------------------")

        print("Spatial Statistics for task: ", task_id)
        print("--------------------")
        print("!!!compared count!!!: ", len(task_spatial_miou_0), len(task_spatial_miou))
        print("!!!spatial evaluation margin=0!!!: ", "miou-margin-0: ", np.mean(task_spatial_miou_0), "thresholded-margin-0: ", np.mean(task_spatial_thresholded_0))
        print("!!!spatial evaluation margin=5!!!: ", "miou-margin-5: ", np.mean(task_spatial_miou_5), "thresholded-margin-5: ", np.mean(task_spatial_thresholded_5))
        print("!!!spatial evaluation margin=10!!!: ", "miou-margin-10: ", np.mean(task_spatial_miou_10), "thresholded-margin-10: ", np.mean(task_spatial_thresholded_10))
        print ("!!!spatial evaluation pairs!!!: ", "miou: ", np.mean(task_spatial_miou), "thresholded: ", np.mean(task_spatial_thresholded))
        print("--------------------")
        filename = "results/spatial_evaluation_" + str(task_id) + ".json"
        with open(filename, "w") as f:
            json.dump({
                "spatial_miou_0": task_spatial_miou_0,
                "spatial_thresholded_0": task_spatial_thresholded_0,
                "spatial_miou_5": task_spatial_miou_5,
                "spatial_thresholded_5": task_spatial_thresholded_5,
                "spatial_miou_10": task_spatial_miou_10,
                "spatial_thresholded_10": task_spatial_thresholded_10,
                "spatial_miou": task_spatial_miou,
                "spatial_thresholded": task_spatial_thresholded,
            }, f, indent=1)
        
    return {
        "spatial_miou_0": np.mean(all_spatial_miou_0),
        "spatial_thresholded_0": np.mean(all_spatial_thresholded_0),
        "spatial_miou_5": np.mean(all_spatial_miou_5),
        "spatial_thresholded_5": np.mean(all_spatial_thresholded_5),
        "spatial_miou_10": np.mean(all_spatial_miou_10),
        "spatial_thresholded_10": np.mean(all_spatial_thresholded_10),
        "spatial_miou": np.mean(all_spatial_miou),
        "spatial_thresholded": np.mean(all_spatial_thresholded),
        "all_spatial_miou_0": all_spatial_miou_0,
        "all_spatial_thresholded_0": all_spatial_thresholded_0,
        "all_spatial_miou_5": all_spatial_miou_5,
        "all_spatial_thresholded_5": all_spatial_thresholded_5,
        "all_spatial_miou_10": all_spatial_miou_10,
        "all_spatial_thresholded_10": all_spatial_thresholded_10,
        "all_spatial_miou": all_spatial_miou,
        "all_spatial_thresholded": all_spatial_thresholded,
        "dataset": evaluated_dataset,
    }

def main():
    run_evaluation_for_task()
    pass

if __name__ == "__main__":
    main()