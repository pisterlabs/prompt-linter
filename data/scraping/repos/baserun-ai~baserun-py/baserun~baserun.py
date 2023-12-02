import inspect
import json
import logging
import traceback
import uuid
from base64 import b64encode, b64decode
from contextlib import contextmanager
from datetime import datetime
from importlib.util import find_spec
from typing import Any, Type, TYPE_CHECKING
from typing import Callable, Dict, Optional, Union

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, _Span, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, get_current_span

from .constants import PARENT_SPAN_NAME
from .evals.evals import Evals
from .exporter import BaserunExporter
from .grpc import get_or_create_submission_service
from .helpers import get_session_id
from .instrumentation.base_instrumentor import BaseInstrumentor
from .instrumentation.span_attributes import SpanAttributes
from .v1.baserun_pb2 import (
    Log,
    SubmitLogRequest,
    Run,
    EndRunRequest,
    TestSuite,
    StartRunRequest,
    EndUser,
    TemplateVersion,
)
from .v1.baserun_pb2_grpc import SubmissionServiceStub

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from baserun.annotation import Annotation


class BaserunEvaluationFailedException(Exception):
    pass


class Baserun:
    _initialized = False

    _instrumentors: list[BaseInstrumentor] = None

    current_test_suite: TestSuite = None
    sessions: dict[str, EndUser] = None

    evals = Evals

    templates: dict[str, TemplateVersion] = None
    used_template_parameters: dict[str, list[dict[str, Any]]] = None

    submission_service: SubmissionServiceStub = None
    async_submission_service: SubmissionServiceStub = None

    @staticmethod
    def init(instrument: bool = True) -> None:
        if Baserun._initialized:
            return

        Baserun._initialized = True
        Baserun.templates = {}
        Baserun.used_template_parameters = {}
        if instrument:
            Baserun.instrument()

    @staticmethod
    def instrument(processor_class: Type[SpanProcessor] = None):
        if not Baserun._instrumentors:
            Baserun._instrumentors = []

        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        if not hasattr(tracer, "active_span_processor"):
            # Check to see if there's an active span processor. If there's not it means that we need to create a new
            # tracer provider and add our span processor to it. (The default type is ProxyTracerProvider which can't
            # have span processors)
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)

        processor_class = processor_class or BatchSpanProcessor
        processor = processor_class(BaserunExporter())
        tracer_provider.add_span_processor(processor)

        if find_spec("openai") is not None:
            from .instrumentation.openai import OpenAIInstrumentor

            instrumentor = OpenAIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            Baserun._instrumentors.append(instrumentor)

        if find_spec("anthropic") is not None:
            from .instrumentation.anthropic import AnthropicInstrumentor

            instrumentor = AnthropicInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            Baserun._instrumentors.append(instrumentor)

    @staticmethod
    def uninstrument():
        for instrumentor in Baserun._instrumentors:
            instrumentor.uninstrument()

        Baserun._instrumentors = []

    @staticmethod
    def serialize_run(run: Run) -> str:
        """Serialize a Run into a base64-encoded string"""
        return b64encode(run.SerializeToString()).decode("utf-8")

    @staticmethod
    def deserialize_run(run_serialized: str) -> Union[Run, None]:
        """Deserialize a base64-encoded string into a Run"""
        if not run_serialized:
            return None

        decoded = b64decode(run_serialized.encode("utf-8"))
        run = Run()
        run.ParseFromString(decoded)
        return run

    @staticmethod
    def current_run() -> Union[Run, None]:
        """Gets the current run"""
        current_span: _Span = get_current_span()
        if not current_span.is_recording():
            return None

        current_run_str = current_span.attributes.get(SpanAttributes.BASERUN_RUN)
        if not current_run_str:
            return None

        return Baserun.deserialize_run(current_run_str)

    @staticmethod
    def _finish_run(run: Run, span: _Span = None):
        try:
            run.completion_timestamp.FromDatetime(datetime.utcnow())
            if span:
                span.set_attribute(
                    SpanAttributes.BASERUN_RUN, Baserun.serialize_run(run)
                )
            get_or_create_submission_service().EndRun.future(EndRunRequest(run=run))
        except Exception as e:
            logger.warning(f"Failed to submit run end to Baserun: {e}")

    @staticmethod
    def get_or_create_current_run(
        name: str = None,
        suite_id: str = None,
        start_timestamp: datetime = None,
        completion_timestamp: datetime = None,
        trace_type: Run.RunType = None,
        metadata: dict[str, Any] = None,
        session_id: str = None,
    ) -> Run:
        """Gets the current run or creates one"""
        existing_run = Baserun.current_run()
        if existing_run:
            return existing_run

        run_id = str(uuid.uuid4())
        if not trace_type:
            trace_type = (
                Run.RunType.RUN_TYPE_TEST
                if Baserun.current_test_suite
                else Run.RunType.RUN_TYPE_PRODUCTION
            )

        if not name:
            raise ValueError("Could not initialize run without a name")

        if not session_id:
            session_id = get_session_id()

        run_data = {
            "run_id": run_id,
            "run_type": trace_type,
            "name": name,
            "metadata": json.dumps(metadata or {}),
            "session_id": session_id,
        }

        if suite_id or Baserun.current_test_suite:
            run_data["suite_id"] = suite_id or Baserun.current_test_suite.id

        run = Run(**run_data)
        run.start_timestamp.FromDatetime(start_timestamp or datetime.utcnow())

        if completion_timestamp:
            run.completion_timestamp.FromDatetime(completion_timestamp)

        try:
            get_or_create_submission_service().StartRun.future(
                StartRunRequest(run=run)
            ).result()
        except Exception as e:
            logger.warning(f"Failed to submit run start to Baserun: {e}")

        return run

    @staticmethod
    def _inputs_from_kwargs(kwargs: dict) -> list[str]:
        inputs = []
        for input_name, input_value in kwargs.items():
            if inspect.iscoroutine(input_value):
                input_result = input_value.__name__
            else:
                input_result = input_value
            inputs.append(f"{input_name}: {input_result}")

        return inputs

    @staticmethod
    def _trace(
        func: Callable,
        run_type: Run.RunType,
        name: str = None,
        metadata: Optional[Dict] = None,
    ) -> Run:
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        run_name = name or func.__name__
        if Baserun.current_test_suite:
            suite_id = Baserun.current_test_suite.id
        else:
            suite_id = None

        if inspect.iscoroutinefunction(func):

            async def wrapper(*args, **kwargs):
                if not Baserun._initialized:
                    return await func(*args, **kwargs)

                session_id = get_session_id()
                run = Baserun.get_or_create_current_run(
                    name=run_name,
                    trace_type=run_type,
                    metadata=metadata,
                    suite_id=suite_id,
                    session_id=session_id,
                )
                with tracer.start_as_current_span(
                    f"{PARENT_SPAN_NAME}.{func.__name__}",
                    kind=SpanKind.CLIENT,
                    attributes={
                        SpanAttributes.BASERUN_RUN: Baserun.serialize_run(run),
                    },
                ) as span:
                    if session_id:
                        span.set_attribute(
                            SpanAttributes.BASERUN_SESSION_ID, session_id
                        )
                    try:
                        result = await func(*args, **kwargs)
                        run.result = str(result) if result is not None else ""
                        return result
                    except Exception as e:
                        run.error = "".join(traceback.format_exception(e))
                        raise e
                    finally:
                        Baserun._finish_run(run, span)

        elif inspect.isasyncgenfunction(func):

            async def wrapper(*args, **kwargs):
                if not Baserun._initialized:
                    async for item in func(*args, **kwargs):
                        yield item

                session_id = get_session_id()
                run = Baserun.get_or_create_current_run(
                    name=run_name,
                    trace_type=run_type,
                    metadata=metadata,
                    suite_id=suite_id,
                    session_id=session_id,
                )

                with tracer.start_as_current_span(
                    f"{PARENT_SPAN_NAME}.{func.__name__}",
                    kind=SpanKind.CLIENT,
                    attributes={
                        SpanAttributes.BASERUN_RUN: Baserun.serialize_run(run),
                    },
                ) as span:
                    if session_id:
                        span.set_attribute(
                            SpanAttributes.BASERUN_SESSION_ID, session_id
                        )

                    try:
                        result = []
                        async for item in func(*args, **kwargs):
                            result.append(item)
                            yield item

                        run.result = str(result) if result is not None else ""
                    except Exception as e:
                        run.error = "".join(traceback.format_exception(e))
                        raise e
                    finally:
                        Baserun._finish_run(run, span)

        else:

            def wrapper(*args, **kwargs):
                if not Baserun._initialized:
                    return func(*args, **kwargs)

                session_id = get_session_id()
                run = Baserun.get_or_create_current_run(
                    name=run_name,
                    trace_type=run_type,
                    metadata=metadata,
                    suite_id=suite_id,
                    session_id=session_id,
                )

                # Create a parent span so we can attach the run to it, all child spans are part of this run.
                with tracer.start_as_current_span(
                    f"{PARENT_SPAN_NAME}.{func.__name__}",
                    kind=SpanKind.CLIENT,
                    attributes={
                        SpanAttributes.BASERUN_RUN: Baserun.serialize_run(run),
                    },
                ) as span:
                    if session_id:
                        span.set_attribute(
                            SpanAttributes.BASERUN_SESSION_ID, session_id
                        )

                    try:
                        result = func(*args, **kwargs)
                        run.result = str(result) if result is not None else ""
                        return result
                    except Exception as e:
                        run.error = "".join(traceback.format_exception(e))
                        raise e
                    finally:
                        Baserun._finish_run(run, span)

        return wrapper

    @staticmethod
    def trace(func: Callable, name: str = None, metadata: Optional[Dict] = None):
        if Baserun.current_test_suite:
            return Baserun.test(func=func, metadata=metadata)

        return Baserun._trace(
            func=func,
            run_type=Run.RunType.RUN_TYPE_PRODUCTION,
            metadata=metadata,
            name=name,
        )

    @staticmethod
    @contextmanager
    def start_trace(*args, name: str = None, **kwargs) -> Run:
        if not Baserun._initialized:
            yield

        # If given a name, ensure that the run has that name. If not, it will only be set when a new run is created
        explicitly_named = name is not None
        if not explicitly_named:
            # stack[0] = start_trace, stack[1] = context manager, stack[2] = user function
            name = inspect.stack()[2].function

        if Baserun.current_test_suite:
            run_type = Run.RunType.RUN_TYPE_TEST
        else:
            run_type = Run.RunType.RUN_TYPE_PRODUCTION

        run = Baserun.get_or_create_current_run(
            name=name,
            trace_type=run_type,
            metadata=kwargs,
        )
        if Baserun.current_test_suite:
            run.suite_id = Baserun.current_test_suite.id

        if explicitly_named:
            run.name = name

        serialized_run = Baserun.serialize_run(run)

        parent_span: _Span = get_current_span()
        if parent_span and parent_span.is_recording():
            parent_span.update_name(f"{PARENT_SPAN_NAME}.{name}")
            parent_span.set_attribute(SpanAttributes.BASERUN_RUN, serialized_run)

        # Create a parent span so we can attach the run to it, all child spans are part of this run.
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        with tracer.start_as_current_span(
            f"{PARENT_SPAN_NAME}.{name}",
            kind=SpanKind.CLIENT,
            attributes={SpanAttributes.BASERUN_RUN: serialized_run},
        ) as span:
            try:
                yield run
            finally:
                Baserun._finish_run(run, span)

    @staticmethod
    def test(func: Callable, metadata: Optional[Dict] = None) -> Run:
        return Baserun._trace(
            func=func, run_type=Run.RunType.RUN_TYPE_TEST, metadata=metadata
        )

    @staticmethod
    def log(name: str, payload: Union[str, Dict]) -> Log:
        if not Baserun._initialized:
            return

        run = Baserun.current_run()
        if not run:
            logger.warning(
                "Cannot send logs to baserun as there is no current trace active."
            )
            return

        log_message = Log(
            run_id=run.run_id,
            name=name,
            payload=json.dumps(payload),
        )
        log_message.timestamp.FromDatetime(datetime.utcnow())
        log_request = SubmitLogRequest(log=log_message, run=run)

        # noinspection PyBroadException
        try:
            get_or_create_submission_service().SubmitLog.future(log_request)
        except Exception as e:
            logger.warning(f"Failed to submit log to Baserun: {e}")

        return log_message

    @staticmethod
    def annotate(
        completion_id: str = None, run: Run = None, trace: Run = None
    ) -> "Annotation":
        """Capture annotations for a particular run and/or completion. the `trace` kwarg here is simply an alias"""
        from baserun.annotation import Annotation

        return Annotation(completion_id=completion_id, run=run or trace)
