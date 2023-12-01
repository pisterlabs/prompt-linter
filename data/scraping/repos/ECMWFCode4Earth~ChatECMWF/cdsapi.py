"""
This module implements the tool for downloading the data from the [CDSAPI](https://cds.climate.copernicus.eu/api-how-to). The amount of [parameters](https://confluence.ecmwf.int/display/CKB/Climate+Data+Store+%28CDS%29+API+Keywords) and different keywords is vaste and diversified, and not documented in a single standard, such as OpenAPI or XML. In this implementation we therefore considered only the reanalysis at single-pressure levels. ERA5 is the fifth generation ECMWF atmospheric reanalysis of the global climate covering the period from January 1940 to present1. It is produced by the Copernicus Climate Change Service (C3S) at ECMWF and provides hourly estimates of a large number of atmospheric, land and oceanic climate variables.
The current version implements the download for a single variable at a time. In this way, it is possible to queue the utility for plotting the variable on the map.

We employ a decorator for entering the function execution and modify the input parameters where needed. In this module in particular, we need to find the correct variable from the list of available ones in the single-pressure reanalysis levels. 
"""
import functools
import json
import os
import subprocess as sp
from typing import Any, Dict, List, Optional
from uuid import uuid4

import cdsapi
import requests
from langchain.tools.base import ToolException
from pydantic import BaseModel, confloat, conint

from ..completion_prompt import create_prompt, get_chatgpt_completion
from ..config import Logger
from ..data_models.era5_variables import variables


def _handle_error(error: ToolException) -> str:
    """
    This is a trial to intercept any tool error and ask the LLM to elaborate on that, eventually improving it.
    It needs to be adjusted together with the maximum amount of iterations the agent will perform for generating an answer.
    """
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
        + "Try to improve on the error correcting the input parameters."
    )


class UserList(BaseModel):
    """
    A list is correctly interpreted within Langchain Structure Tool only if supplied as __root__ attribute, and not directly in the function signature.
    """

    __root__: List[int]


def create_request(
    variable,
    year: List[str] = ["2016"],
    month: List[str] = ["01"],
    day: List[str] = ["01", "02"],
    time: List[str] = [
        "00:00",
        "01:00",
        "02:00",
        "03:00",
        "04:00",
        "05:00",
        "06:00",
        "07:00",
        "08:00",
        "09:00",
        "10:00",
        "11:00",
        "12:00",
        "13:00",
        "14:00",
        "15:00",
        "16:00",
        "17:00",
        "18:00",
        "19:00",
        "20:00",
        "21:00",
        "22:00",
        "23:00",
    ],
    area: List[int] = [
        47,
        5,
        35,
        18,
    ],
) -> Dict[str, str]:
    """
    This function wraps the input in the json for the CDSAPI request submission.
    The CDSAPI requests string for all the inputs.

    Args:
    -----
       variable: str
       year: List[str]
             formatted YYYY
       month: List[str]
             formatted mm
       day: List[str]
             formatted dd
       time: List[str]
             formatted HH:MM
       area: List[int]
             list of 4 elements, in the following order, [lat_max, lon_min, lat_min, lon_max]
    Returns:
    --------
        request: Dict[str, str]
                the JSON for the CDSAPI request
    """
    if time is None or time == []:
        time = [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ]
    request = {
        "product_type": "reanalysis",
        "variable": variable,
        "year": year,
        "month": month,
        "day": day,
        "time": time,
        "area": area,
        "format": "grib",
    }
    return request


def validate_inputs(func):
    """
    Function wrapper asking ChatGPT to choose a good match for the input variable of the function, in the way expressed in the user query.
    """

    @functools.wraps(func)
    def wrapper_validation(*args, **kwargs):
        Logger.debug(f"args: {args}, kwargs: {kwargs}")
        product_prompt = create_prompt(kwargs["variable"], variables)
        response = get_chatgpt_completion(product_prompt)
        response = response.replace("'", "")
        Logger.info(f"the matched product: {response}")
        kwargs["variable"] = response
        output = func(*args, **kwargs)
        return output

    return wrapper_validation


@validate_inputs
def cdsapi_tool(
    variable: str,
    year: UserList,
    month: UserList,
    day: UserList,
    lat_min: confloat(ge=-90.0, le=90.0),
    lat_max: confloat(ge=-90.0, le=90.0),
    lon_min: confloat(ge=-180.0, le=180.0),
    lon_max: confloat(ge=-180.0, le=180.0),
    hours: Optional[UserList] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Use this tool when asked for climate data services datasets.

    Args:
    -----
       variable: str
       year: UserList,
       month: UserList,
       day: UserList,
       lat_min: confloat(ge=-90., le=90.),
       lat_max: confloat(ge=-90., le=90.),
       lon_min: confloat(ge=-180., le=180.),
       lon_max: confloat(ge=-180., le=180.),
       hours: Optional[UserList] = None,
    Returns:
    --------
        request: Dict[str, str]
                the JSON for the CDSAPI request
    Raises:
    -------
        ToolException
    """
    c = cdsapi.Client()

    year = [str(y) for y in year]
    month = [str(m).zfill(2) for m in month]
    day = [str(d).zfill(2) for d in day]
    if hours is not None:
        hours = [f"{h:02d}:00" for h in hours]
    request = create_request(
        variable, year, month, day, hours, [lat_max, lon_min, lat_min, lon_max]
    )
    map_uuid = uuid4()
    if not os.path.exists(f"./data/{map_uuid}"):
        os.makedirs(f"./data/{map_uuid}")
    try:
        result = c.retrieve(
            "reanalysis-era5-single-levels", request, f"./data/{map_uuid}/download.grib"
        )
        success = None
    except Exception as e:
        return json.dumps({"result": e.msg, "success": False, "request": request})
    if result:
        try:
            sp.run(
                [
                    "python3",
                    "read_grib_info.py",
                    "-f",
                    f"../../data/{map_uuid}/download.grib",
                ],
                cwd="./src/grib",
            )
            success = True
        except:
            success = False
    else:
        Logger.error("Something went wrong retrieving the data...")
        raise ToolException("Something went wrong employing the tool")

    return json.dumps({"result": f"{map_uuid}", "success": success, "request": request})
