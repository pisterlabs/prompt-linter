from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.trackee import Trackee

from . import OpenaiTrackees


class _FineTuningTrackees:
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None], method_name: str):
        fine_tuning_jobs = Trackee(
            obj=client.fine_tuning.jobs if client else openai.fine_tuning.jobs,
            method_name=method_name,
            span_name=f"fine_tuning.jobs.{method_name}",
        )

        fine_tuning_raw_jobs = Trackee(
            obj=client.fine_tuning.with_raw_response.jobs
            if client
            else openai.fine_tuning.with_raw_response.jobs,
            method_name=method_name,
            span_name=f"fine_tuning.with_raw_response.jobs.{method_name}",
        )

        fine_tuning_jobs_raw = Trackee(
            obj=client.fine_tuning.jobs.with_raw_response
            if client
            else openai.fine_tuning.jobs.with_raw_response,
            method_name=method_name,
            span_name=f"fine_tuning.jobs.with_raw_response.{method_name}",
        )

        self.trackees = [
            fine_tuning_jobs,
            fine_tuning_raw_jobs,
            fine_tuning_jobs_raw,
        ]

        if client:
            raw_fine_tuning = Trackee(
                obj=client.with_raw_response.fine_tuning.jobs,
                method_name=method_name,
                span_name=f"with_raw_response.fine_tuning.jobs.{method_name}",
            )
            self.trackees.append(raw_fine_tuning)


class _FineTuningListTrackees(_FineTuningTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="list")


class _FineTuningCreateTrackees(_FineTuningTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="create")


class _FineTuningCancelTrackees(_FineTuningTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="cancel")


class _FineTuningRetrieveTrackees(_FineTuningTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="retrieve")


class _FineTuningListEventsTrackees(_FineTuningTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="list_events")


class FineTuningTrackees(OpenaiTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        list_trackees = _FineTuningListTrackees(client=client)
        create_trackees = _FineTuningCreateTrackees(client=client)
        cancel_trackees = _FineTuningCancelTrackees(client=client)
        retrieve_trackees = _FineTuningRetrieveTrackees(client=client)
        list_events_trackees = _FineTuningListEventsTrackees(client=client)

        trackees = (
            list_trackees.trackees
            + create_trackees.trackees
            + cancel_trackees.trackees
            + retrieve_trackees.trackees
            + list_events_trackees.trackees
        )

        super().__init__(trackees=trackees, client=client)
