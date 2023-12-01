import os
import uuid
from typing import Optional, Dict, Any

from langchain.callbacks import LangChainTracer
from langchain.chains.base import Chain
from langchain.schema.runnable import RunnableConfig, RunnableSequence
from langchain.schema.runnable.base import Input
from langchain.schema.runnable.config import ensure_config

from xpuls.mlmonitor.langchain.handlers.callback_handlers import CallbackHandler
from xpuls.mlmonitor.langchain.patches.utils import get_scoped_override_labels, get_scoped_project_info
from xpuls.mlmonitor.langchain.profiling.prometheus import LangchainPrometheusMetrics
from xpuls.mlmonitor.langchain.xpuls_client import XpulsAILangChainClient


def patch_invoke(ln_metrics: LangchainPrometheusMetrics, xpuls_client: XpulsAILangChainClient):
    # Store the original run method

    runnable_invoke = RunnableSequence.invoke
    runnable_ainvoke = RunnableSequence.ainvoke
    chain_invoke = Chain.invoke
    chain_ainvoke = Chain.ainvoke

    def _apply_patch(input: Input, config: Optional[RunnableConfig] = None):
        override_labels = get_scoped_override_labels()
        project_details = get_scoped_project_info()
        updated_labels = dict(ln_metrics.get_default_labels(), **override_labels)
        chain_run_id = str(uuid.uuid4())

        ln_tracer = LangChainTracer(
            project_name=project_details['project_id'] if project_details['project_id'] is not None else
            project_details['project_slug'],
            client=xpuls_client,
        )

        callback_handler = CallbackHandler(ln_metrics, chain_run_id, override_labels)

        updated_config = ensure_config(config)

        with ln_metrics.agent_run_histogram.labels(**dict(ln_metrics.get_default_labels(), **override_labels)).time():
            if updated_config.get("callbacks") is not None:
                updated_config['callbacks'].append(callback_handler)
            else:
                updated_config['callbacks'] = [callback_handler]

            if os.getenv("XPULSAI_TRACING_ENABLED", "false") == "true":
                updated_config['callbacks'].append(ln_tracer)
            metadata = {'xpuls': {'labels': updated_labels, 'run_id': chain_run_id,
                                  'project_id': project_details['project_id'] if project_details[
                                                                                     'project_id'] is not None else
                                  project_details['project_slug']}}

            updated_config['metadata'] = dict(updated_config['metadata'], **metadata)

        return updated_config, updated_labels

    def patched_chain_invoke(self, input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,):
        updated_config, updated_labels = _apply_patch(input=input, config=config)
        # Call the original run method
        return chain_invoke(self, input, updated_config, **kwargs)

    async def patched_chain_ainvoke(self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,):
        updated_config, updated_labels = _apply_patch(input=input, config=config)

        # Call the original run method
        return chain_ainvoke(self, input, updated_config, **kwargs)

    def patched_runnable_invoke(self, input: Input, config: Optional[RunnableConfig] = None):
        updated_config, updated_labels = _apply_patch(input=input, config=config)
        # Call the original run method
        return runnable_invoke(self, input, updated_config)

    async def patched_runnable_ainvoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs):
        updated_config, updated_labels = _apply_patch(input=input, config=config)

        # Call the original run method
        return runnable_ainvoke(self, input, updated_config, **kwargs)

    # Patch the Chain class's invoke method with the new one
    Chain.invoke = patched_chain_invoke
    Chain.ainvoke = patched_chain_ainvoke

    # Patch the RunnableSequence class's invoke method with the new one
    RunnableSequence.invoke = patched_runnable_invoke
    RunnableSequence.ainvoke = patched_runnable_ainvoke
