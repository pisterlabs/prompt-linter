"""
The kernels channel in RTU is primarily used for runtime updates like kernel and cell status,
variable explorer, and outputs vice document model changes on the files channel (adding cells,
updating content, etc)
"""
import uuid
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from origami.models.kernels import CellState, KernelStatusUpdate
from origami.models.rtu.base import BaseRTURequest, BaseRTUResponse, BooleanReplyData


class KernelsRequest(BaseRTURequest):
    channel_prefix: Literal["kernels"] = "kernels"


class KernelsResponse(BaseRTUResponse):
    channel_prefix: Literal["kernels"] = "kernels"


class KernelSubscribeRequestData(BaseModel):
    file_id: uuid.UUID


class KernelSubscribeRequest(KernelsRequest):
    event: Literal["subscribe_request"] = "subscribe_request"
    data: KernelSubscribeRequestData


# Kernel status is returned on subscribe and also updated through kernel status updates
class KernelSubscribeReplyData(BaseModel):
    success: bool
    kernel_session: Optional[KernelStatusUpdate] = None  # None if no Kernel is alive for a file


class KernelSubscribeReply(KernelsResponse):
    event: Literal["subscribe_reply"] = "subscribe_reply"
    data: KernelSubscribeReplyData


class KernelStatusUpdateResponse(KernelsResponse):
    event: Literal["kernel_status_update_event"] = "kernel_status_update_event"
    data: KernelStatusUpdate


# Cell State
class BulkCellStateUpdateData(BaseModel):
    cell_states: List[CellState]


class BulkCellStateUpdateResponse(KernelsResponse):
    event: Literal["bulk_cell_state_update_event"] = "bulk_cell_state_update_event"
    data: BulkCellStateUpdateData


# Variable explorer updates return a list of current variables in the kernel
# On connect to a new Kernel, Clients can send a request to trigger an event. Otherwise events occur
# after cell execution automatically.
class VariableExplorerUpdateRequest(KernelsRequest):
    event: Literal["variable_explorer_update_request"] = "variable_explorer_update_request"


# It is confusing but variable_explorer_update_request can either be an RTU client to Gate server
# (RTURequest) or also be propogated out by Gate from another client, meaning it comes in as a
# server-to-client (RTUResponse) so we need to model it just to avoid warning about unmodeled msgs
class VariableExplorerUpdateRequestPropogated(KernelsResponse):
    event: Literal["variable_explorer_update_request"] = "variable_explorer_update_request"
    data: dict = Field(default_factory=dict)


class VariableExplorerResponse(KernelsResponse):
    event: Literal["variable_explorer_event"] = "variable_explorer_event"


class IntegratedAIRequestData(BaseModel):
    prompt: str
    # this may not be called on a specific cell, but at a specific point in time at a generic
    # "document" level, so we don't require a cell_id
    cell_id: Optional[str] = None
    # if a cell_id is provided and this is True, the result will be added to the cell's output
    # instead of just sent back as an RTU reply
    output_for_response: bool = False


class IntegratedAIRequest(KernelsRequest):
    event: Literal["integrated_ai_request"] = "integrated_ai_request"
    data: IntegratedAIRequestData


class IntegratedAIReply(KernelsResponse):
    event: Literal["integrated_ai_reply"] = "integrated_ai_reply"
    data: BooleanReplyData


class IntegratedAIEvent(KernelsResponse):
    event: Literal["integrated_ai_event"] = "integrated_ai_event"
    # same data as the IntegratedAIRequest, just echoed back out
    data: IntegratedAIRequestData


class IntegratedAIResultData(BaseModel):
    # the full response from OpenAI; in most cases, sidecar will have either created a new cell
    # or an output, so this result should really only be used when the RTU client needs it to exist
    # outside of the cell/output structure
    result: str


# this is sidecar to gate as a result of calling the OpenAIHandler method (OpenAI response,
# error, etc); after that, Gate propogates the data out as an IntegratedAIEvent
class IntegratedAIResult(KernelsRequest):
    event: Literal["integrated_ai_result"] = "integrated_ai_result"
    data: IntegratedAIResultData


class IntegratedAIResultReply(KernelsResponse):
    event: Literal["integrated_ai_result_reply"] = "integrated_ai_result_reply"
    data: BooleanReplyData


class IntegratedAIResultEvent(KernelsResponse):
    event: Literal["integrated_ai_result_event"] = "integrated_ai_result_event"
    data: IntegratedAIResultData


KernelRequests = Annotated[
    Union[
        KernelSubscribeRequest,
        VariableExplorerUpdateRequest,
        IntegratedAIRequest,
        IntegratedAIResult,
    ],
    Field(discriminator="event"),
]

KernelResponses = Annotated[
    Union[
        KernelSubscribeReply,
        KernelStatusUpdateResponse,
        BulkCellStateUpdateResponse,
        VariableExplorerUpdateRequestPropogated,
        VariableExplorerResponse,
        IntegratedAIReply,
        IntegratedAIResultReply,
        IntegratedAIEvent,
        IntegratedAIResultEvent,
    ],
    Field(discriminator="event"),
]
