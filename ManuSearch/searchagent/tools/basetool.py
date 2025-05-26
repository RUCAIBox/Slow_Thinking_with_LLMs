from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BaseTool(ABC, BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

    def __call__(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return self.execute(**kwargs)

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""

    def to_schema(self) -> Dict:
        """Convert tool to function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            },
        }
