from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class RawProviderResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_id: str
    stage: str
    attempt: int
    provider: str
    model: str
    content: str
    sha256: str

    @classmethod
    def capture(cls, *, content: str, **metadata) -> "RawProviderResponse":
        return cls(
            **metadata,
            content=content,
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )


def write_raw_response_jsonl(response: RawProviderResponse, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(response.model_dump_json() + "\n")
        handle.flush()
