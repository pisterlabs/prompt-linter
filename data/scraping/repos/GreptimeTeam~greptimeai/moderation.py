from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.trackee import Trackee

from . import OpenaiTrackees


class ModerationTrackees(OpenaiTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        moderations_create = Trackee(
            obj=client.moderations if client else openai.moderations,
            method_name="create",
            span_name="moderations.create",
        )

        moderations_raw_create = Trackee(
            obj=client.moderations.with_raw_response
            if client
            else openai.moderations.with_raw_response,
            method_name="create",
            span_name="moderations.with_raw_response.create",
        )

        trackees = [
            moderations_create,
            moderations_raw_create,
        ]

        if client:
            raw_moderations_create = Trackee(
                obj=client.with_raw_response.moderations,
                method_name="create",
                span_name="with_raw_response.moderations.create",
            )
            trackees.append(raw_moderations_create)


        super().__init__(trackees=trackees, client=client)
