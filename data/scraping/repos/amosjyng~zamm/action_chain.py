"""Chain that chooses and performs the next action."""
from typing import Any, Dict, List, Union

from langchain_contrib.chains import ChoiceChain

from zamm.actions.base import Action

from .choice.base import ChoicePickerChain


class ActionChain(ChoiceChain):
    choice_picker: ChoicePickerChain
    actions: Dict[str, Action]
    step_output_key: str = "step_output"
    _should_trace: bool = False

    def __init__(self, actions: List[Action], **kwargs):
        actions_dict = {action.name: action for action in actions}
        choices = {action.name: action.chain for action in actions}
        super().__init__(actions=actions_dict, choices=choices, **kwargs)

    @property
    def output_keys(self) -> List[str]:
        """Possible output keys produced by this chain."""
        return super().output_keys + [self.step_output_key]

    def __call__(
        self,
        inputs: Union[Dict[str, Any], Any],
        return_only_outputs: bool = False,
    ) -> Dict[str, Any]:
        output = super().__call__(inputs, return_only_outputs)
        action = self.actions[output["choice"]]
        output[self.step_output_key] = action.output_type.from_chain_output(output)
        return output
