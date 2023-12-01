from typing import List, ClassVar

from ghoshell.container import Provider
from ghoshell.framework.bootstrapper import FileLoggerBootstrapper, \
    CommandFocusDriverBootstrapper, LLMToolsFocusDriverBootstrapper
from ghoshell.framework.ghost import GhostKernel
from ghoshell.llms import LLMTextCompletion, OpenAIChatCompletion
from ghoshell.llms.openai import OpenAIBootstrapper
from ghoshell.llms.thinks import ConversationalThinksBootstrapper, FileAgentMindsetBootstrapper
from ghoshell.mocks.ghost_mock.bootstrappers import *
from ghoshell.mocks.providers import *
from ghoshell.prototypes.playground.llm_test_ghost import GameUndercoverBootstrapper
from ghoshell.prototypes.playground.llm_test_ghost import LLMConversationalThinkBootstrapper, \
    PromptUnitTestsBootstrapper


class MockGhost(GhostKernel):
    # 启动流程. 想用这种方式解耦掉系统文件读取等逻辑.

    bootstrapper: ClassVar[List] = [
        FileLoggerBootstrapper(),
        RegisterThinkDemosBootstrapper(),
        CommandFocusDriverBootstrapper(),
        OpenAIBootstrapper(),

        # 使用 llm chat completion 实现的思维
        ConversationalThinksBootstrapper(),
        # 使用 llm chat completion + function call 实现的思维.
        FileAgentMindsetBootstrapper(),

        # deprecated:
        LLMConversationalThinkBootstrapper(),
        LLMToolsFocusDriverBootstrapper(),
        # 将 configs/llms/unitests 下的文件当成单元测试思维.
        PromptUnitTestsBootstrapper(),
        # 测试加入 undercover 游戏. deprecated
        GameUndercoverBootstrapper(think_name="game/undercover"),
    ]

    depending_contracts: ClassVar[List] = [
        LLMTextCompletion,
        OpenAIChatCompletion,
    ]

    contracts_providers: ClassVar[List] = [
        MockCacheProvider(),
        MockAPIRepositoryProvider(),
        MockOperationKernelProvider(),
        MockThinkMetaDriverProvider(),
    ]

    def get_bootstrapper(self) -> List[GhostBootstrapper]:
        return self.bootstrapper

    def get_depending_contracts(self) -> List:
        contracts = super().get_depending_contracts()
        contracts += self.depending_contracts
        return contracts

    def get_contracts_providers(self) -> List[Provider]:
        return self.contracts_providers
