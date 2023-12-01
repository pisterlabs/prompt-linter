import os
import ast
import json

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from langchain.chains import LLMChain

from LangChainPipeline.PromptTemplates.all_parameters_prompt import get_all_parameters_prompt_chat as get_all_parameters_prompt
from LangChainPipeline.PromptTemplates.text_content_prompt import get_text_content_prompt_chat as get_text_content_prompt
from LangChainPipeline.PromptTemplates.image_query_prompt import get_image_query_prompt_chat as get_image_query_prompt

from LangChainPipeline.PydanticClasses.EditParameters import EditParameters

from LangChainPipeline.DataFilters.semantic_filters import filter_metadata_by_semantic_similarity
from LangChainPipeline.utils import timecode_to_seconds

from backend.image_retriever import get_first_google_search_image

class EditChain():
    def __init__(
        self,
        verbose=False,
        top_k = 10,
        neighbors_left = 0,
        neighbors_right = 0,
        video_id="4LdIvyfzoGY",
        interval=10
    ):
        self.visual_metadata = None
        self.transcript_metadata = None
        self.interval = None
        self.video_id = None
        self.set_video(video_id, interval)
        
        self.all_parameters = AllParametersChain(
            verbose=verbose,
        )

        self.text_content = TextContentChain(
            verbose=verbose,
            top_k=top_k,
            neighbors_left=neighbors_left,
            neighbors_right=neighbors_right,
        )

        self.image_query = ImageQueryChain(
            verbose=verbose,
            top_k=top_k,
            neighbors_left=neighbors_left,
            neighbors_right=neighbors_right,
        )

        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Initialized EditChain")

    def set_video(self, video_id, interval):
        if self.video_id == video_id and self.interval == interval:
            return

        self.video_id = video_id

        self.is_combined = True
        metadata_filepath = f"metadata/{video_id}_{str(interval)}_combined.txt"
        if os.path.exists(metadata_filepath) == False:
            print("ERROR: Metadata file does not exist: ", metadata_filepath)
            metadata_filepath = f"metadata/{video_id}_{str(interval)}.txt"
            self.is_combined = False

        self.visual_metadata = []
        self.transcript_metadata = []
        with open(metadata_filepath) as f:
            raw_lines = f.readlines()
            for line in raw_lines:
                interval = ast.literal_eval(line.rstrip())
                visual_data = {
                    "action": interval["action_pred"],
                    "abstract_caption": interval["synth_caption"],
                    "dense_caption": interval["dense_caption"],
                }
                visual_data_str = (interval["synth_caption"].strip() + ", " 
                    + interval["dense_caption"].strip() + ", " 
                    + interval["action_pred"].strip())
                
                if self.is_combined:
                    visual_data = {
                        "action": interval["action_pred"],
                        "abstract_caption": interval["synth_caption"],
                        "objects": interval["objects"],
                        "dense_caption": interval["dense_caption_2"],
                    }
                    visual_data_str = (json.dumps(interval["synth_caption"]).strip() + ", "
                        + json.dumps(interval["dense_caption_2"]).strip() + ", "
                        + json.dumps(interval["action_pred"]).strip() + ", "
                        + json.dumps(interval["objects"]).strip())

                transcript = interval["transcript"].strip()
                self.visual_metadata.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "data": visual_data_str,
                    "structured_data": visual_data,
                })
                self.transcript_metadata.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "data": transcript,
                })
        print("Set video EditChain")

    def set_parameters(self, top_k, neighbors_left, neighbors_right):
        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        self.text_content.set_parameters(top_k, neighbors_left, neighbors_right)
        self.image_query.set_parameters(top_k, neighbors_left, neighbors_right)
        print("Set parameters")

    def run_all_parameters(self,
        original_command, 
        parameters, initial_edit_parameters,
        video_shape
    ):
        context = [
            f'The original command was: {original_command}',
            f'Video Properties are: height: {str(video_shape[0])}, width: {str(video_shape[1])}'
        ]
        new_edit_parameters = self.all_parameters.run(context, parameters, initial_edit_parameters)
        return new_edit_parameters

    def run_text_content(
        self,
        original_command,
        parameters,
        initial_edit_parameters,
        start, finish,
    ):
        context = [
            f'The original command was: {original_command}',
        ]
        metadata_transcript = list(filter(lambda x: start <= timecode_to_seconds(x["start"]) < finish, self.transcript_metadata))
        metadata_visual = list(filter(lambda x: start <= timecode_to_seconds(x["start"]) < finish, self.visual_metadata))
        if len(parameters["textParameters"]) > 0:
            initial_edit_parameters["textParameters"]["content"] = self.text_content.run(
                context,
                metadata_transcript,
                metadata_visual,
                parameters["textParameters"],
            )
        return initial_edit_parameters

    def run_image_query(self,
        original_command,
        parameters,
        initial_edit_parameters,
        start, finish,
    ):
        context = [
            f'The original command was: {original_command}',
        ]
        metadata_transcript = list(filter(lambda x: start <= timecode_to_seconds(x["start"]) < finish, self.transcript_metadata))
        metadata_visual = list(filter(lambda x: start <= timecode_to_seconds(x["start"]) < finish, self.visual_metadata))
        if len(parameters["imageParameters"]) > 0:
            initial_edit_parameters["imageParameters"]["searchQuery"] = self.image_query.run(
                context,
                metadata_transcript,
                metadata_visual,
                parameters["imageParameters"],
            )

            initial_edit_parameters["imageParameters"]["source"] = get_first_google_search_image(
                initial_edit_parameters["imageParameters"]["searchQuery"],
                initial_edit_parameters["imageParameters"]["source"]
            )
        return initial_edit_parameters
    
    def run_crop_parameters(self,
        spatial_parameters,
        initial_edit_parameters,
        video_shape,
    ):
        initial_edit_parameters["cropParameters"]["x"] = 0
        initial_edit_parameters["cropParameters"]["y"] = 0
        initial_edit_parameters["cropParameters"]["width"] = video_shape[1]
        initial_edit_parameters["cropParameters"]["height"] = video_shape[0]
        initial_edit_parameters["cropParameters"]["cropX"] = spatial_parameters["x"]
        initial_edit_parameters["cropParameters"]["cropY"] = spatial_parameters["y"]
        initial_edit_parameters["cropParameters"]["cropWidth"] = spatial_parameters["width"]
        initial_edit_parameters["cropParameters"]["cropHeight"] = spatial_parameters["height"]
        return initial_edit_parameters



class AllParametersChain():
    def __init__(
            self,
            verbose=False,
    ):
        self.skip_parameters = ["imageParameters", "cutParameters", "cropParameters"]
        self.llm = ChatOpenAI(temperature=0.1, model_name="gpt-4-1106-preview")
        self.parser = PydanticOutputParser(pydantic_object=EditParameters)

        self.prompt_template = get_all_parameters_prompt({
            "format_instructions": self.parser.get_format_instructions(),
        })

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=verbose,
            output_parser=self.parser,
        )
        print("Initialized AllParametersChain")

    def filter_parameters(self, parameters):
        filtered_parameters = {}
        for parameter in parameters:
            if parameter not in self.skip_parameters:
                filtered_parameters[parameter] = parameters[parameter]
        return filtered_parameters

    def run(self, context, parameters, initial_edit_parameters):
        filtered_parameters = self.filter_parameters(parameters)
        filtered_edit_parameters = self.filter_parameters(initial_edit_parameters)

        total_references = 0
        for parameter in filtered_parameters:
            total_references += len(filtered_parameters[parameter])
        
        if total_references == 0:
            return initial_edit_parameters
        try:
            # #dummy
            # return initial_edit_parameters

            result = self.chain.predict(
                context=json.dumps(context),
                command=json.dumps(filtered_parameters),
                initial_parameters=json.dumps(filtered_edit_parameters),
            )
        except:
            print("ERROR: Failed to adjust parameters: ", filtered_parameters)
            return initial_edit_parameters
        dict_result = result.dict()
        for parameter in self.skip_parameters:
            dict_result[parameter] = initial_edit_parameters[parameter]
        return dict_result
    
class TextContentChain():
    def __init__(
        self,
        verbose=False,
        top_k = 10,
        neighbors_left = 0,
        neighbors_right = 0,
    ):
        self.llm = ChatOpenAI(temperature=0.1, model_name="gpt-4-1106-preview")

        self.prompt_template = get_text_content_prompt()

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=verbose,
        )

        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Initialized TextContentChain")

    def set_parameters(self, top_k, neighbors_left, neighbors_right):
        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Set parameters")

    def run(self, context, metadata_transcript, metadata_visual, command):

        filtered_metadata_transcript = filter_metadata_by_semantic_similarity(
            targets=command,
            candidates=metadata_transcript,
            k=self.top_k/2,
            neighbors_left=self.neighbors_left,
            neighbors_right=self.neighbors_right,
        )
        filtered_metadata_visual = filter_metadata_by_semantic_similarity(
            targets=command,
            candidates=metadata_visual,
            k=self.top_k/2,
            neighbors_left=self.neighbors_left,
            neighbors_right=self.neighbors_right,
        )

        try: 
            result = self.chain.predict(
                context=json.dumps(context),
                metadata_transcript=json.dumps([data["data"] for data in filtered_metadata_transcript]),
                metadata_visual=json.dumps([data["structured_data"] for data in filtered_metadata_visual]),
                command=json.dumps(command),
            )
        except:
            print("ERROR: Failed to adjust text content: ", command)
            return ""
        return result
    
class ImageQueryChain():
    def __init__(
        self,
        verbose=False,
        top_k = 10,
        neighbors_left = 0,
        neighbors_right = 0,
    ):
        self.llm = ChatOpenAI(temperature=0.1, model_name="gpt-4-1106-preview")

        self.prompt_template = get_image_query_prompt()

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=verbose,
        )

        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Initialized ImageQueryChain")

    def set_parameters(self, top_k, neighbors_left, neighbors_right):
        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Set parameters")

    ### TODO: need to consider only metadata from the relevant segment
    def run(self, context, metadata_transcript, metadata_visual, command):
        filtered_metadata_transcript = filter_metadata_by_semantic_similarity(
            targets=command,
            candidates=metadata_transcript,
            k=self.top_k/2,
            neighbors_left=self.neighbors_left,
            neighbors_right=self.neighbors_right,
        )
        filtered_metadata_visual = filter_metadata_by_semantic_similarity(
            targets=command,
            candidates=metadata_visual,
            k=self.top_k/2,
            neighbors_left=self.neighbors_left,
            neighbors_right=self.neighbors_right,
        )

        try:
            result = self.chain.predict(
                context=json.dumps(context),
                metadata_transcript=json.dumps([data["data"] for data in filtered_metadata_transcript]),
                metadata_visual=json.dumps([data["structured_data"] for data in filtered_metadata_visual]),
                command=json.dumps(command),
            )
        except:
            print("ERROR: Failed to adjust image query: ", command)
            return ""
        return result