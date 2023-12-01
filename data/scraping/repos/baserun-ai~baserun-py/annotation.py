import json
import logging
from contextlib import contextmanager, asynccontextmanager
from numbers import Number
from typing import Any, Union

from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import get_current_span, Span

from baserun import Baserun
from baserun.constants import PARENT_SPAN_NAME
from baserun.grpc import (
    get_or_create_submission_service,
    get_or_create_async_submission_service,
)
from baserun.v1.baserun_pb2 import (
    Log,
    Check,
    Feedback,
    Run,
    SubmitAnnotationsRequest,
    CompletionAnnotations,
)

logger = logging.getLogger(__name__)


class Annotation:
    completion_id: str
    span: Span
    logs: list[Log]
    checks: list[Check]
    feedback_list: list[Feedback]

    def __init__(self, completion_id: str = None, run: Run = None):
        self.run = run or Baserun.get_or_create_current_run()
        self.span = self.try_get_span()
        self.completion_id = completion_id
        self.logs = []
        self.checks = []
        self.feedback_list = []

    @classmethod
    def annotate(
        cls, completion: Union[None, ChatCompletion, Stream[ChatCompletionChunk]] = None
    ):
        completion_id = completion.id if completion else None
        return cls(completion_id=completion_id)

    @classmethod
    @asynccontextmanager
    async def aanotate(
        cls, completion: Union[None, ChatCompletion, Stream[ChatCompletionChunk]] = None
    ):
        if not Baserun._initialized:
            yield

        annotation = cls.annotate(completion=completion)
        try:
            yield annotation
        finally:
            try:
                await annotation.asubmit()
            except BaseException as e:
                logger.warning(f"Could not submit annotation to baserun: {e}")

    @classmethod
    @contextmanager
    def annotate(
        cls, completion: Union[None, ChatCompletion, Stream[ChatCompletionChunk]] = None
    ):
        if not Baserun._initialized:
            yield

        annotation = cls.annotate(completion=completion)
        try:
            yield annotation
        finally:
            try:
                annotation.submit()
            except BaseException as e:
                logger.warning(f"Could not submit annotation to baserun: {e}")

    def feedback(
        self,
        name: str = None,
        thumbsup: bool = None,
        stars: Number = None,
        score: Number = None,
        metadata: dict[str, Any] = None,
    ):
        if score is None:
            if thumbsup is not None:
                score = 1 if thumbsup else 0
            elif stars is not None:
                score = stars / 5
            else:
                logger.info(
                    "Could not calculate feedback score, please pass a score, thumbsup, or stars"
                )
                score = 0

        run = Baserun.get_or_create_current_run()
        feedback_kwargs = {"name": name or "General Feedback", "score": score}
        if metadata:
            feedback_kwargs["metadata"] = json.dumps(metadata)

        if run.session_id:
            end_user = Baserun.sessions.get(run.session_id)
            if end_user:
                feedback_kwargs["end_user"] = end_user

        feedback = Feedback(**feedback_kwargs)
        self.feedback_list.append(feedback)

    def check(
        self,
        name: str,
        methodology: str,
        expected: dict[str, Any],
        actual: dict[str, Any],
        score: Number = None,
        metadata: dict[str, Any] = None,
    ):
        check = Check(
            name=name,
            methodology=methodology,
            actual=json.dumps(actual),
            expected=json.dumps(expected),
            score=score or 0,
            metadata=json.dumps(metadata or {}),
        )
        self.checks.append(check)

    def check_includes(
        self,
        name: str,
        expected: Union[str, list[str]],
        actual: str,
        metadata: dict[str, Any] = None,
    ):
        expected_list = [expected] if isinstance(expected, str) else expected
        result = any(expected in actual for expected in expected_list)
        return self.check(
            name=name,
            methodology="includes",
            expected={"value": expected},
            actual={"value": actual},
            score=1 if result else 0,
            metadata=metadata,
        )

    def log(self, name: str, metadata: dict[str, Any]):
        log = Log(
            run_id=self.run.run_id,
            name=name,
            payload=json.dumps(metadata),
        )
        self.logs.append(log)

    def try_get_span(self) -> Span:
        current_span: ReadableSpan = get_current_span()
        if current_span and current_span.name.startswith(f"{PARENT_SPAN_NAME}."):
            return current_span

        # TODO? Maybe we should create a span or trace
        return None

    def submit(self):
        annotation_message = CompletionAnnotations(
            completion_id=self.completion_id,
            checks=self.checks,
            logs=self.logs,
            feedback=self.feedback_list,
        )
        get_or_create_submission_service().SubmitAnnotations.future(
            SubmitAnnotationsRequest(annotations=annotation_message, run=self.run)
        )

    async def asubmit(self):
        annotation_message = CompletionAnnotations(
            completion_id=self.completion_id,
            checks=self.checks,
            logs=self.logs,
            feedback=self.feedback_list,
        )

        await get_or_create_async_submission_service().SubmitAnnotations(
            SubmitAnnotationsRequest(annotations=annotation_message, run=self.run)
        )
