import os
import uuid
from typing import Dict, Any, Optional

from langchain.callbacks import LangChainTracer
from langchain.chains.base import Chain
from langchain.schema.runnable import RunnableConfig, RunnableSequence
from langsmith import Client

from xpuls.mlmonitor.langchain.decorators.telemetry_override_labels import TelemetryOverrideLabels
from xpuls.mlmonitor.langchain.decorators.map_xpuls_project import MapXpulsProject
from xpuls.mlmonitor.langchain.handlers.callback_handlers import CallbackHandler
from xpuls.mlmonitor.langchain.patches.utils import get_scoped_project_info, get_scoped_override_labels

from xpuls.mlmonitor.langchain.profiling.prometheus import LangchainPrometheusMetrics
from xpuls.mlmonitor.langchain.xpuls_client import XpulsAILangChainClient


def patch_run(ln_metrics: LangchainPrometheusMetrics, xpuls_client: XpulsAILangChainClient):
    # Store the original run method
    original_run = Chain.run
    original_arun = Chain.arun

    def _apply_patch(kwargs,):
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

        with ln_metrics.agent_run_histogram.labels(**dict(ln_metrics.get_default_labels(), **override_labels)).time():
            if 'callbacks' in kwargs:
                kwargs['callbacks'].append(callback_handler)
            else:
                kwargs['callbacks'] = [callback_handler]

            if os.getenv("XPULSAI_TRACING_ENABLED", "false") == "true":
                kwargs['callbacks'].append(ln_tracer)
            metadata = {'xpuls': {'labels': updated_labels, 'run_id': chain_run_id,
                                  'project_id': project_details['project_id'] if project_details[
                                                                                     'project_id'] is not None else
                                  project_details['project_slug']}}
            if 'metadata' in kwargs:
                kwargs['metadata'] = dict(kwargs['metadata'], **metadata)
            else:
                kwargs['metadata'] = metadata
        return kwargs, ln_tracer, updated_labels

    def patched_run(self, *args, **kwargs):

        updated_kwargs, ln_tracer, updated_labels = _apply_patch(kwargs)

        # Call the original run method
        return original_run(self, *args, **updated_kwargs)

    async def patched_arun(self, *args, **kwargs):
        updated_kwargs, ln_tracer, updated_labels = _apply_patch(kwargs)

        # Call the original run method
        return original_arun(self, *args, **updated_kwargs)

    # Patch the Chain class's run method with the new one
    Chain.run = patched_run
    Chain.arun = patched_arun


