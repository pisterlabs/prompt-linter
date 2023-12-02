from __future__ import annotations


from typing import Any, List
from agent.toolkits.git_integrator.base import create_git_integration_toolkit
from agent.toolkits.git_integrator.prompt import GIT_AGENT_DESCRIPTION
from agent.toolkits.git_integrator.toolkit import GitIntegratorToolkit
from agent.toolkits.gitlab_integration.prompt import GITLAB_AGENT_DESCRIPTION
from agent.toolkits.gitlab_integration.toolkit import GitlabIntegrationToolkit
from agent.toolkits.k8s_explorer.base import create_k8s_explorer_agent
from agent.toolkits.k8s_explorer.prompt import K8S_EXPLORER_AGENT_DESCRIPTION
from agent.toolkits.k8s_explorer.toolkit import K8sExplorerToolkit

from langchain.llms.base import BaseLLM
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from langchain.agents.tools import Tool
from langchain.callbacks.base import BaseCallbackManager
from agent.toolkits.k8s_sme.base import create_k8s_sme_agent
from agent.toolkits.k8s_sme.prompt import K8S_SME_AGENT_DESCRIPTION
from agent.toolkits.k8s_sme.toolkit import KubernetesSMEToolkit

from tools.git_integrator.tool import GitModel
from tools.gitlab_integration.tool import GitlabModel
from tools.k8s_explorer.tool import KubernetesOpsModel
from tools.k8s_sme.tools import KubernetesSMEModel
from tools.slack_integration.tool import SlackModel, SlackSendMessageTool


class K8sEngineerToolkit(BaseToolkit):
    """Toolkit for performing engineering tasks related to kubernetes."""

    git_agent: AgentExecutor
    gitlab_agent: AgentExecutor
    k8s_explorer_agent: AgentExecutor
    k8s_sme_agent: AgentExecutor
    slack_tool: BaseTool

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        # git_agent_tool = Tool(
        #     name="git_integration_agent",
        #     func=self.git_agent.run,
        #     description=GIT_AGENT_DESCRIPTION,
        # )
        k8s_explorer_agent_tool = Tool(
            name="k8s_explorer_agent",
            func=self.k8s_explorer_agent.run,
            description=K8S_EXPLORER_AGENT_DESCRIPTION,
        )
        gitlab_agent_tool = Tool(
            name="gitlab_agent",
            func=self.gitlab_agent.run,
            description=GITLAB_AGENT_DESCRIPTION,
        )
        k8s_sme_agent_tool = Tool(
            name="k8s_sme_agent",
            func=self.k8s_sme_agent.run,
            description=K8S_SME_AGENT_DESCRIPTION,
        )
        return [k8s_explorer_agent_tool, gitlab_agent_tool, k8s_sme_agent_tool, self.slack_tool]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        k8s_model: KubernetesOpsModel,
        git_model: GitModel,
        gitlab_model: GitlabModel,
        slack_model: SlackModel,
        k8s_sme_model: KubernetesSMEModel,
        callback_manager: BaseCallbackManager = None,
        slack_channel: str = None,
        slack_thread_ts: str = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> K8sEngineerToolkit:
        """Create a toolkit from an LLM."""
        git_agent = create_git_integration_toolkit(
            llm=llm, toolkit=GitIntegratorToolkit(model=git_model, callback_manager=callback_manager), verbose=verbose, callback_manager=callback_manager, **kwargs)
        k8s_explorer_agent = create_k8s_explorer_agent(
            llm=llm, toolkit=K8sExplorerToolkit(model=k8s_model, callback_manager=callback_manager), verbose=verbose, callback_manager=callback_manager, **kwargs)
        gitlab_agent = create_git_integration_toolkit(
            llm=llm, toolkit=GitlabIntegrationToolkit(model=gitlab_model, callback_manager=callback_manager), verbose=verbose, callback_manager=callback_manager, **kwargs)
        k8s_sme_agent = create_k8s_sme_agent(
            llm=llm, toolkit=KubernetesSMEToolkit(model=k8s_sme_model, callback_manager=callback_manager), verbose=verbose, callback_manager=callback_manager, **kwargs)
        slack_tool = SlackSendMessageTool(model = slack_model, channel=slack_channel, thread_ts=slack_thread_ts, verbose=verbose, callback_manager=callback_manager, **kwargs)
        return cls(git_agent=git_agent, k8s_explorer_agent=k8s_explorer_agent, gitlab_agent=gitlab_agent, k8s_sme_agent=k8s_sme_agent, slack_tool=slack_tool, **kwargs)
