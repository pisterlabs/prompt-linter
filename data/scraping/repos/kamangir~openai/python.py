from typing import List
import random
from typing import Dict, Any, Tuple
from openai_cli.completion.api import complete_prompt
from openai_cli.completion.functions.generic import ai_function
import cv2
import skimage
from abcli.modules.host import is_jupyter
from abcli.modules import objects
from abcli import file
import numpy as np
from abcli.logging import crash_report
import abcli.logging
import logging

logger = logging.getLogger()


class ai_function_py(ai_function):
    def __init__(
        self,
        output_class_name: str = "",
        verbose=None,
        validation_input=None,
    ):
        super().__init__(verbose=verbose)

        self.language = "python"
        self.output_class_name = output_class_name
        self.validation_input = validation_input

        self.function_handle = None

    def compute(self, inputs):
        if self.function_handle is None:
            return None

        logger.info(
            "{}.compute({})".format(
                self.__class__.__name__,
                inputs.__class__.__name__,
            )
        )

        outputs = self.function_handle(inputs)
        logger.info("-> {}".format(outputs.__class__.__name__))

        if self.output_class_name:
            if outputs.__class__.__name__ != self.output_class_name:
                logger.error(
                    "{}.compute({}): {} != {}".format(
                        self.__class__.__name__,
                        inputs.__class__.__name__,
                        outputs.__class__.__name__,
                        self.output_class_name,
                    )
                )
                return None

        return outputs

    def generate(
        self,
        prompt,
        retry: int = 5,
    ) -> Tuple[bool, Dict[str, Any]]:
        success, metadata = super().generate(prompt)
        if not success:
            return success, metadata

        self.function_handle = None

        try:
            exec(self.code)
        except:
            crash_report("{}.generate() failed.".format(self.__class__.__name__))

        for thing in locals().values():
            if hasattr(thing, "__name__") and thing.__name__ == self.function_name:
                self.function_handle = thing
                break

        passed = self.function_handle is not None

        if passed and self.validation_input is not None:
            try:
                validation_output = self.compute(self.validation_input)

                if validation_output is None:
                    passed = False
            except:
                crash_report("compute test failed.")
                passed = False

        if passed:
            if self.auto_save:
                self.save()
            return True, metadata

        if retry <= 0:
            logger.info("last retry, failed.")
            return False, metadata

        logger.info("{} more tries.".format(retry))
        return self.generate(prompt, retry - 1)

    def to_json(self):
        output = super().to_json()
        output.update({"output_class_name": self.output_class_name})
        return output
