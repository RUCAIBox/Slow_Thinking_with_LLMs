from typing import Any, Optional
from ..tools.basetool import BaseTool

class FinalAnswerTool(BaseTool):

    name:str = "final_answer"
    description:str = "Provides a final answer to the given problem."
    parameters: Optional[dict] = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The final answer to the problem"
            }
        },
        "required": [
            "answer"
        ],
        "additionalProperties": False
    }

    def execute(self, answer:Any) -> Any:
        return answer