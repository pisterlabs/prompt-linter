import json
import re
from typing import Dict, Optional, Any

import html2text
from langchain.utils import get_from_dict_or_env
from pydantic import BaseModel, root_validator


class NicovideoSnapshotApiWrapper(BaseModel):
    nicovideo_client: Any  #: :meta private:
    nicovideo_field_type: Any
    nicovideo_agent_name: Optional[str] = None

    def run(self, query: str) -> str:
        valid_query = re.match(r"Search query: (.+)", query)
        if valid_query is None or len(valid_query.groups()) == 0:
            return "Be sure to use the prefix 'Search query: '."
        response = (
            self.nicovideo_client.keywords()
            .single_query(valid_query.groups()[0])
            .field(
                {
                    self.nicovideo_field_type.TITLE,
                    self.nicovideo_field_type.DESCRIPTION,
                    self.nicovideo_field_type.VIEW_COUNTER,
                }
            )
            .sort(self.nicovideo_field_type.VIEW_COUNTER)
            .no_filter()
            .limit(20)
            .user_agent(self.nicovideo_agent_name, "0")
            .request()
        )
        if response.status()[0] == 200:
            data = response.data()
            if len(data) == 0:
                return "Video does not exist or search results are inappropriate."
            return json.dumps(
                list(map(_shortening_description, data)), ensure_ascii=False
            )
        elif response.status()[0] == 400:
            return "Bad Request query parse error"
        elif response.status()[0] == 500:
            return "Internal Server Error. please retry later."
        elif response.status()[0] == 503:
            return "Service Unavailable (MAINTENANCE). please retry later."

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        nicovideo_agent_name = get_from_dict_or_env(
            values, "nicovideo_agent_name", "NICOVIDEO_AGENT_NAME"
        )
        values["nicovideo_agent_name"] = nicovideo_agent_name

        try:
            from nicovideo_api_client.api.v2.snapshot_search_api_v2 import (
                SnapshotSearchAPIV2,
            )
            from nicovideo_api_client.constants import FieldType
        except ImportError:
            raise ImportError(
                "nicovideo_api_client is not installed. "
                "Please install it with `pip install nicovideo_api_client`"
            )
        client = SnapshotSearchAPIV2
        nicovideo_field_type = FieldType
        values["nicovideo_client"] = client
        values["nicovideo_field_type"] = nicovideo_field_type
        return values


def _shortening_description(video: Dict[str, Any]) -> Dict[str, Any]:
    video["description"] = html2text.html2text(video["description"])[:20]
    return video
