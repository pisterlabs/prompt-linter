from typing import List, Tuple, Callable
import React
from react_i18next import useTranslation
from utils.trpc import trpc
from types.openai import OpenAIModels
from types.prompt import Prompt
from pages.api.home.home_context import HomeContext
from uuid import uuid4

PromptsAction = {
    "update": Callable[[Prompt], Promise[List[Prompt]]],
    "updateAll": Callable[[List[Prompt]], Promise[List[Prompt]]],
    "add": Callable[[], Promise[List[Prompt]]],
    "remove": Callable[[Prompt], Promise[List[Prompt]]]
}

def usePrompts() -> Tuple[List[Prompt], PromptsAction]:
    """
    Custom hook for managing prompts.

    Returns:
        A tuple containing the list of prompts and the prompt actions.
    """
    tErr = useTranslation('error')['t']
    homeContext = useContext(HomeContext)
    defaultModelId = homeContext['state']['defaultModelId']
    prompts = homeContext['state']['prompts']
    dispatch = homeContext['dispatch']
    promptsUpdateAll = trpc.prompts.updateAll.useMutation()
    promptsUpdate = trpc.prompts.update.useMutation()
    promptRemove = trpc.prompts.remove.useMutation()

    def updateAll(updated: List[Prompt]) -> Promise[List[Prompt]]:
        """
        Updates all prompts.

        Args:
            updated: The updated list of prompts.

        Returns:
            The updated list of prompts.
        """
        promptsUpdateAll.mutateAsync(updated)
        dispatch({ 'field': 'prompts', 'value': updated })
        return updated

    def add() -> Promise[List[Prompt]]:
        """
        Adds a new prompt.

        Returns:
            The updated list of prompts.
        """
        if not defaultModelId:
            err = tErr('No Default Model')
            raise Error(err)
        newPrompt: Prompt = {
            'id': uuid4(),
            'name': f"Prompt {len(prompts) + 1}",
            'description': '',
            'content': '',
            'model': OpenAIModels[defaultModelId],
            'folderId': None
        }
        promptsUpdate.mutateAsync(newPrompt)
        newState = [newPrompt] + prompts
        dispatch({ 'field': 'prompts', 'value': newState })
        return newState

    def update(prompt: Prompt) -> Promise[List[Prompt]]:
        """
        Updates a specific prompt.

        Args:
            prompt: The updated prompt.

        Returns:
            The updated list of prompts.
        """
        newState = [prompt if f['id'] == prompt['id'] else f for f in prompts]
        promptsUpdate.mutateAsync(prompt)
        dispatch({ 'field': 'prompts', 'value': newState })
        return newState

    def remove(prompt: Prompt) -> Promise[List[Prompt]]:
        """
        Removes a specific prompt.

        Args:
            prompt: The prompt to be removed.

        Returns:
            The updated list of prompts.
        """
        newState = [f for f in prompts if f['id'] != prompt['id']]
        promptRemove.mutateAsync({ 'id': prompt['id'] })
        dispatch({ 'field': 'prompts', 'value': newState })
        return newState

    return prompts, {
        'add': add,
        'update': update,
        'updateAll': updateAll,
        'remove': remove
    }
