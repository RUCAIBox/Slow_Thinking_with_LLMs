from ..tools.basetool import BaseTool
from ..models.searcher import Searcher
from typing import Optional

class SolvePlan(BaseTool):

    name: str = "SolvePlan"
    description: str = "A tool that solve the subproblems made by planner."
    parameters: Optional[dict] = {
        "type":"object",
        "properties":{
            "plans":{                    
                "type":"array",
                "items":{
                    "type":"string"
                },
                "description": "The plans need to be solved."
            }
        },
        "required":[
            "plans"
        ],
        "additionalProperties":False
    }
    searcher: Searcher
    model_config = {
        "arbitrary_types_allowed": True
    }

    def __init__(self, 
        searcher: Searcher 
    ):

        super().__init__(searcher=searcher)

    def execute(self, *, plans, recorder):

        if isinstance(plans, str):
            plans = [plans]

        plan_answer = []
        
        for plan in plans:


            gen = self.searcher.search(question=plan, recorder=recorder)
            try:
                while True:
                    tool_name, content, references_url = next(gen)
                    if tool_name == 'webpages':
                        yield {
                            'status':'webpages',
                            'content':content
                        }
                    elif tool_name == "model_response":
                        yield {
                            'status': 'model_think',
                            'content': content,
                        }
                        
                    else:

                        yield {
                            'status': 'searching',
                            'substatus': tool_name,
                            'tool_return': content,
                            'ref2url': references_url,
                            'current_node_name': plan
                        }
                    
            except StopIteration as e:
                    plan_answer.append(
                        f" {plan} :\n {e.value}"
                    )
                
        return plan_answer