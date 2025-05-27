import copy, uuid, json, logging, traceback
from collections import defaultdict
from typing import Dict, List
from collections import defaultdict
from ..utils.utils import *
from ..schema import AgentMessage 
logging.basicConfig(level=logging.INFO)
logging.getLogger("watchdog").setLevel(logging.INFO)



class WebSearchGraph:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, str]] = {}  # Dictionary to store all nodes in the graph. Each node is indexed by its name and contains content, type, and other relevant information.
        self.adjacency_list: Dict[str, List[dict]] = defaultdict(list)  # Adjacency list to store connections between nodes. Each node is indexed by its name and contains a list of adjacent node names.

    def add_root_node(
        self,
        node_content: str,
        node_name: str = "root",
    ):
        """Add the root node to the graph.

        Args:
            node_content (str): Content of the node.
            node_name (str, optional): Name of the node. Defaults to 'root'.
        """
        self.nodes[node_name] = dict(content=node_content, type="root")
        self.adjacency_list[node_name] = []
    
    def add_node(
        self,
        node_name: str,
        node_content: str,
    ):
        """Add a search sub-question node.

        Args:
            node_name (str): Name of the node.
            node_content (str): Content of the sub-question.

        Returns:
            str: Returns the search result.
        """
        self.nodes[node_name] = dict(content=node_content, type="searcher")
        self.adjacency_list[node_name] = []

        # Get parent nodes to retrieve historical conversation context.
        parent_nodes = []
        for start_node, adj in self.adjacency_list.items():
            for neighbor in adj:
                if (
                    node_name == neighbor
                    and start_node in self.nodes
                    and "response" in self.nodes[start_node]
                ):
                    parent_nodes.append(self.nodes[start_node])

        parent_response = [
            dict(question=node['content'], answer=node['response']) for node in parent_nodes
        ]

        self.nodes[node_name]["response"] = None
        self.nodes[node_name]["memory"] = []
        self.nodes[node_name]["session_id"] = None

    def add_response_node(self, node_name="response"):
        """Add a response node.
        If the current information meets the query requirements, add a response node.

        Args:
            thought (str): Reasoning process.
            node_name (str, optional): Name of the node. Defaults to 'response'.
        """
        self.nodes[node_name] = dict(type="end")
    
    def add_edge(self, start_node: str, end_node: str):
        """Add an edge between two nodes.

        Args:
            start_node (str): Name of the starting node.
            end_node (str): Name of the ending node.
        """
        self.adjacency_list[start_node].append(dict(id=str(uuid.uuid4()), name=end_node, state=2))

    def reset(self):
        """Reset the graph by clearing all nodes and edges."""
        self.nodes = {}
        self.adjacency_list = defaultdict(list)
    
    def node(self, node_name: str) -> str:
        """Retrieve a node by its name.

        Args:
            node_name (str): Name of the node.

        Returns:
            str: A copy of the node's data.
        """
        return self.nodes[node_name].copy()

class Recorder:
    """ Records information from multiple steps of the query process. """
    def __init__(self, action):
        """
        Initializes the Recorder object to track query content and memory.

        Args:
            action (`str`): The action to be performed by the recorder.
        """
        self.action = action
        self.container = dict()
        self.container['content'] = WebSearchGraph() # 查询图
        self.container['memory'] = dict() # 记录每个模块的memory  
        self.container['memory']['searcher'] = []

    def _construct_graph(self,message):
        if isinstance(message, str):
            nodes = [message]
        elif isinstance(message, dict):
            nodes = []
            for v in message.values():
                if isinstance(v, list):
                    nodes.extend(v)
                elif isinstance(v, str):
                    nodes.append(v)
                else:
                    raise ValueError('UNSUPPORTED DATA TYPE')
        elif isinstance(message, list):
            nodes = message
        else:
            raise ValueError('UNSUPPORTED DATA TYPE')
        if nodes is None:
            nodes = []
        for node in nodes:
            self.container['content'].add_node(
                node_name=node, # 节点名称（对应着plan）
                node_content=None # 子问题内容（对应着searcher中生成的多步query）
            )
        return nodes


    def update(self, node_name, node_content, content, memory, sender):
        """
        Updates the content and memory for a given node based on the sender type.

        Args:
            node_name (`str`): The name of the node being updated.
            node_content(`str`): The content of the node being updated.
            content (`str`): The content to store in the node.
            memory (`dict`): The memory to store for the node.
            sender (`str`): The sender module (e.g., 'planner', 'searcher', 'reader').
        """
        if sender == 'planner':
            new_node = self._construct_graph(content)
            self.container['memory']['planner'] = memory 
            return new_node
        
        elif sender == 'searcher':
            self.container['content'].nodes[node_name]['content'] = node_content
            if isinstance(content, str):
                ref2url = {int(k): v for k, v in json.loads(content).items()}
                self.container['content'].nodes[node_name]['memory'] = ref2url
            elif isinstance(content, dict):
                ref2url = {int(k): v for k, v in content.items()}
                self.container['content'].nodes[node_name]['memory'] = ref2url
            else:
                raise ValueError("content must be instance of string or dict")
        
        elif sender == 'searcher_response':
            self.container['content'].nodes[node_name]['response'] = content
            if memory:
                self.container['memory']['searcher'].append(copy.deepcopy(memory))
        elif sender == 'reasoner':
            self.container['content'].add_response_node()


    def generate_reason_process(self):
        
        graph  = self.container['content']
        
        reason_process = copy.deepcopy(graph.nodes)
        count = 0
        for subquery in reason_process.keys():
            if subquery not in ['root', 'response']:
                cache_memory = []
                if len(self.container['memory']['searcher']) > count:
                    for cache in self.container['memory']['searcher'][count].get_memory():
                        if isinstance(cache, AgentMessage):
                            if isinstance(cache.content, str):
                                cache_memory.append(cache.content)
                            else:
                                tool_calls = []
                                for message_tool_call in cache.content.tool_calls:
                                    tool_calls.append({"id":message_tool_call.id, "arguments": message_tool_call.function.arguments, "name": message_tool_call.function.name})
                                cache_memory.append(tool_calls)
                        else:
                            
                            cache_memory.append(cache)
                
                reason_process[subquery]['searcher'] = cache_memory
                count += 1

        return reason_process
        # return json.dumps(reason_process, ensure_ascii=False, indent=2)