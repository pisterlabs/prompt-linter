import os

from loguru import logger
from openai_multi_client import OpenAIMultiClient

from grants_tagger_light.augmentation.augment_openai import AugmentOpenAI


class ParallelAugmentOpenAI(AugmentOpenAI):
    def __init__(self, prompt_template_path, model_key="gpt-3.5-turbo"):
        super().__init__(prompt_template_path, model_key)
        self.api = OpenAIMultiClient(
            endpoint="chats", data_template={"model": self.model_key}
        )

    @staticmethod
    def _process_response(result):
        if result.failed:
            logger.warning(
                f"Failed to get augmentation for {result.metadata['featured_tag']}"
            )
            return
        choices = result.response["choices"]
        AugmentOpenAI.process_choices(choices, result.metadata)

    def _make_requests(
        self,
        collect_concurrent_calls,
        dset,
        temperature,
        top_p,
        presence_penalty,
        num_proc,
        model_key,
        save_to_path,
    ):
        for num in range(len(collect_concurrent_calls)):
            tag = collect_concurrent_calls[num][0]
            missing_num = collect_concurrent_calls[num][1]

            for data, metadata in self._prepare_request(
                tag,
                missing_num,
                dset,
                temperature,
                top_p,
                presence_penalty,
                num_proc,
                model_key,
                save_to_path,
            ):
                self.api.request(
                    data=data, metadata=metadata, callback=self._process_response
                )

    def generate(
        self,
        collect_concurrent_calls,
        dset,
        save_to_path,
        model_key,
        temperature=1.5,
        top_p=1,
        presence_penalty=0,
        num_proc=os.cpu_count(),
    ):
        self.api.run_request_function(
            self._make_requests,
            collect_concurrent_calls=collect_concurrent_calls,
            dset=dset,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            num_proc=num_proc,
            model_key=model_key,
            save_to_path=save_to_path,
        )

        self.api.pull_all()
