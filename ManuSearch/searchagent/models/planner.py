from typing import Union, List
from ..models.basellm import BaseStreamingAgent
from ..schema import AgentMessage 
from ..utils.utils import *
from ..utils.utils import *


# The planner agent is responsible for generating the overall plan based on user input
class Planner(BaseStreamingAgent):
    """
    Planner class that organizes the overall search and reasoning steps.
    Responsible for breaking down the user query into subqueries and planning the approach.
    """  
    def __init__(
        self,
        **baseconfig
    ):

        super().__init__(**baseconfig)

    def plan(self, message: Union[List[Union[AgentMessage,str]],Union[AgentMessage,str]], recorder, session_id=0, **kwargs):
        """
        Generates the current plan and subqueries based on the user input and recorder.
        
        Args:
            message (`Union[List[Union[AgentMessage,str]],Union[AgentMessage,str]]`): The input message.
            recorder (`Recorder`): The recorder that tracks the state.
            session_id (`int`, optional): The session ID for the current interaction.
        
        Returns:
            current_plan (`AgentMessage`): The plan for the current step.
            current_subquerys (`list`): The subqueries to be processed.
        """
        
        if isinstance(message, list):
            for m in message[:-1]:
                self.agent.update_memory(m)

            for response in super().forward(message[-1],session_id=session_id, **kwargs):
                yield response
            
        else:
            for response in super().forward(message,session_id=session_id, **kwargs):
                yield response

        response = parse_resp_to_json(response.content)
        
        if response.get('actions', '').strip().lower() == 'extract_problems':
            current_subquerys = recorder.update(
                node_name=None,
                node_content=None,
                content=response['content'],
                memory=self.agent.memory,
                sender='planner'
            )
        elif response.get('actions', '').strip().lower() == 'final_response':
            recorder.update(
                node_name = None,
                node_content=None,
                content=response['content'],
                memory=self.agent.memory,
                sender='reasoner'
            )

        return 
    

