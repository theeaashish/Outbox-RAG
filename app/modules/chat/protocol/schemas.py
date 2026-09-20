from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AGUIMessageInput(BaseModel):
    """Single message entry in AG-UI RunAgentInput."""

    id: str | None = None
    role: str
    content: str
    name: str | None = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class RunAgentInput(BaseModel):
    """AG-UI RunAgentInput request wire contract."""

    thread_id: str | None = Field(default=None, alias="threadId")
    run_id: str | None = Field(default=None, alias="runId")
    parent_run_id: str | None = Field(default=None, alias="parentRunId")
    state: Any = None
    messages: list[AGUIMessageInput] = Field(default_factory=list)
    tools: list[Any] = Field(default_factory=list)
    context: list[Any] = Field(default_factory=list)
    forwarded_props: dict[str, Any] | None = Field(default=None, alias="forwardedProps")

    model_config = ConfigDict(populate_by_name=True, extra="allow")
