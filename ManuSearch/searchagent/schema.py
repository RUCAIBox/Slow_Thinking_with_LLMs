from enum import IntEnum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
import os, sys
    
# need to integrate int, so asdict can convert AgentStatusCode to int
class ModelStatusCode(IntEnum):
    END = 0  # end of streaming
    STREAM_ING = 1  # response is in streaming
    SERVER_ERR = -1  # triton server's error
    SESSION_CLOSED = -2  # session has been closed
    SESSION_OUT_OF_LIMIT = -3  # request length out of limit
    SESSION_INVALID_ARG = -4  # invalid argument
    SESSION_READY = 2  # session is ready for inference


class AgentStatusCode(IntEnum):
    END = 0  # end of streaming
    STREAM_ING = 1  # response is in streaming
    SERVER_ERR = -1  # triton server's error
    SESSION_CLOSED = -2  # session has been closed
    SESSION_OUT_OF_LIMIT = -3  # request length out of limit
    SESSION_INVALID_ARG = -4  # invalid argument
    SESSION_READY = 2  # session is ready for inference
    PLUGIN_START = 3  # start tool
    PLUGIN_END = 4  # finish tool
    PLUGIN_RETURN = 5  # finish tool
    CODING = 6  # start python
    CODE_END = 7  # end python
    CODE_RETURN = 8  # python return


class AgentMessage(BaseModel):
    content: Any
    sender: str = 'user'
    formatted: Optional[Any] = None
    extra_info: Optional[Any] = None
    type: Optional[str] = None
    receiver: Optional[str] = None
    stream_state: Union[ModelStatusCode, AgentStatusCode] = AgentStatusCode.END