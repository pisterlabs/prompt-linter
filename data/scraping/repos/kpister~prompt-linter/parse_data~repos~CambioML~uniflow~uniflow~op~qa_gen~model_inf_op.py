"""Model inference operation."""
import copy
import logging
from typing import Any, Mapping

from openai import OpenAI

from uniflow.op.basic.linear_op import LinearOp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelInfOp(LinearOp):
    """Model inference operation.

    Args:
        nodes (Sequence[Node]): Input nodes.

    Returns:
        Sequence[Node]: Output nodes.
    """

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Call the language model to generate outputs for the prompt.
        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        logger.info("Starting ModelInfOp...")
        qaa_list_encoded = copy.deepcopy(value_dict["qaa_list_encoded"])
        qaa_augmented_raw = []
        client = OpenAI()
        for i, batch_inputs_string in enumerate(qaa_list_encoded):
            logger.info(f"Running batch {i + 1} of {len(qaa_list_encoded)}...")
            completion_batch = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": batch_inputs_string},
                ],
                model="gpt-3.5-turbo",
                temperature=0.2,
                max_tokens=1000,  # The maximum number of tokens to generate in the completion
            )
            results_string = completion_batch.choices[0].message.content
            qaa_augmented_raw.append(results_string)
        logger.info("ModelInfOp complete!")
        return {"qaa_augmented_raw": qaa_augmented_raw}
