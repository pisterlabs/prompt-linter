'''
The `.torch.obervation.block` module wraps the widget and engine components in a MemoryBlock.
'''

from src.rl.showcase.interface import Display
from typing import Any, Generic, Sequence, TypeVar

from result.result import Err, Ok, Result
from src.rl.blocks.interface import ObservationBlock
from typing_extensions import TypeAlias
from .engine import OpenAI_Observation, OpenAI_ObservationFactory

import streamlit as st

S = TypeVar('S')
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


Parent: TypeAlias = Any
Children: TypeAlias = Sequence[Display[Any, Any, Any]]

Engine: TypeAlias = OpenAI_Observation[S, T, U, V]
Factory: TypeAlias = OpenAI_ObservationFactory[S, T, U, V]
Block: TypeAlias = ObservationBlock[S, T, U, V, Parent, Children, Engine]

class OpenAI_ObservationBlock(Generic[S, T, U, V], Block[S, T, U, V]):
    '''
    Block for displaying an OpenAI_Observation in a ObservationBlock.
    '''

    make = Factory[S, T, U, V]

    __engine__ : Engine[S, T, U, V]


    def __init__(self, engine: Engine[S, T, U, V], **kwargs: Any) -> None:
        '''
        Wraps an OpenAI_Observation in a block.
        '''
        self.__engine__ = engine

    def engine(self, **kwargs: Any) -> Result[Engine[S, T, U, V], ValueError]:
        '''
        Returns an OpenAI_Observation instance.
        '''
        return Ok(self.__engine__)

    def display(self, parent: Parent, children: Children, **kwargs: Any) -> Result[Any, ValueError]:
        '''
        Displays an OpenAI_Observation.
        '''
        try:
            parent.text(f'current reward: {self.engine().unwrap().get_reward()}')

            return Ok(None)

        except ValueError as error: return Err(ValueError(error, 'displaying OpenAI_Observation failed'))

    def update(self, parent: Parent, children: Children, **kwargs: Any) -> Result[Any, ValueError]:
        '''
        Updates an OpenAI_ObservationBlock.
        '''
        try: return Ok(None)

        except ValueError as error: return Err(ValueError(error, 'displaying OpenAI_Observation failed'))

    