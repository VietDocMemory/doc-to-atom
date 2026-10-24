"""Credential-key recognition shared by config fingerprints and artifact writers."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REDACTED = "<redacted>"
# Match complete credential names, optionally qualified by a service/environment
# prefix (e.g. OPENAI_API_KEY). Token credentials use explicit names: arbitrary
# *_token fields may be vocabulary values, including multimodal/custom tokens.
_CREDENTIAL_NAMES = frozenset({
    "api_key", "apikey", "password", "passwd", "secret", "secrets",
    "credential", "credentials", "authorization", "private_key", "secret_key",
    "access_key", "access_key_id", "api_token", "access_token", "refresh_token",
    "auth_token", "authentication_token", "bearer_token", "session_token", "id_token",
    "oauth_token", "client_token", "hf_token", "huggingface_token", "github_token",
    "gitlab_token", "gh_token", "wandb_token", "openai_token", "minimax_token",
})


def is_credential_key(key: object) -> bool:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key))
    name = re.sub(r"[-.\s]+", "_", name).lower().strip("_")
    return name == "token" or any(
        name == credential or name.endswith("_" + credential)
        for credential in _CREDENTIAL_NAMES
    )


def redact_secrets(value: Any, *, cli_arguments: bool = True) -> Any:
    """Copy metadata without interpreting semantic lists as command lines.

    A standalone sequence or a mapping's command/argv field is a command line;
    other nested sequences (e.g. tokenizer vocabulary) retain their values.
    """
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if is_credential_key(key) else redact_secrets(
                item, cli_arguments=str(key).lower() in {"command", "argv"}
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact_secrets(list(value), cli_arguments=cli_arguments))
    if isinstance(value, list):
        result = []
        redact_next = False
        for item in value:
            if redact_next:
                result.append(REDACTED)
                redact_next = False
            elif cli_arguments and isinstance(item, str) and item.startswith("-"):
                flag, equals, _argument = item.partition("=")
                if is_credential_key(flag):
                    result.append(f"{flag}={REDACTED}" if equals else flag)
                    redact_next = not equals
                else:
                    result.append(item)
            else:
                result.append(redact_secrets(item, cli_arguments=False))
        return result
    return value
