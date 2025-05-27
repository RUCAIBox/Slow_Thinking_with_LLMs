"""Collection classes for managing multiple tools."""
from typing import Any, Dict, List

from ..tools.basetool import BaseTool


class ToolCollection:
    """A collection of defined tools."""

    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def __iter__(self):
        return iter(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        return [tool.to_param() for tool in self.tools]

    def execute(
        self, *, name: str, tool_input: Dict[str, Any] = None
    ) -> Any:
        tool = self.tool_map.get(name)
        if 'required' in tool_input:
            tool_input.pop('required')
        if not tool:
            raise TypeError(f"Tool {name} is invalid")
        try:
            result = tool(**tool_input)
            return result
        except:
            raise ValueError(f"Tool {name} can not execute")

    def get_tool(self, name: str) -> BaseTool:
        return self.tool_map.get(name)

    def add_tool(self, tool: BaseTool):
        self.tools += (tool,)
        self.tool_map[tool.name] = tool
        return self

    def add_tools(self, *tools: BaseTool):
        for tool in tools:
            self.add_tool(tool)
        return self