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
        self.nodes: Dict[str, Dict[str,str]] = {} # 存储图中所有节点的字典。每个节点由其名称索引，并包含内容、类型以及其他相关信息。
        self.adjacency_list: Dict[str, List[dict]] = defaultdict(list) # 存储图中所有节点之间连接关系的邻接表。每个节点由其名称索引，并包含一个相邻节点名称的列表。

    def add_root_node(
        self,
        node_content : str,
        node_name: str = "root",
    ):
        """添加起始节点

        Args:
            node_content (str): 节点内容
            node_name (str, optional): 节点名称. Defaults to 'root'.

        """
        self.nodes[node_name] = dict(content=node_content, type="root")
        self.adjacency_list[node_name] = []
    
    def add_node(
        self,
        node_name: str,
        node_content: str,
    ):
        """添加搜索子问题节点

        Args:
            node_name (str): 节点名称
            node_content (str): 子问题内容

        Returns:
            str: 返回搜索结果
        """

        self.nodes[node_name] = dict(content=node_content, type="searcher")
        self.adjacency_list[node_name] = []

        # 获取父节点，以获得历史对话信息
        parent_nodes = []
        for start_node, adj in self.adjacency_list.items():
            for neighbor in adj:
                if(
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
        """添加回复节点
        如果当前获取的信息已经满足问题需求，添加回复节点。

        Args:
            thought (str): 思考过程
            node_name (str, optional): 节点名称. Defaults to 'response'.

        """
        self.nodes[node_name] = dict(type="end")
    
    def add_edge(self, start_node:str, end_node:str):
        """添加边
        Args:
            start_node(str) : 起始节点名称
            end_node(str): 结束节点名称
        """
        self.adjacency_list[start_node].append(dict(id=str(uuid.uuid4()), name=end_node, state=2))


    def reset(self):
        self.nodes = {}
        self.adjacency_list = defaultdict(list)
    
    def node(self, node_name:str) -> str:
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

        return json.dumps(reason_process, ensure_ascii=False, indent=2)