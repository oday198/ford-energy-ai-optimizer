import os
import inspect
from typing import Optional

from langfuse import Langfuse


def get_langfuse_client() -> Optional[Langfuse]:
    pub = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    sec = os.getenv("LANGFUSE_SECRET_KEY", "").strip()

    host = os.getenv("LANGFUSE_HOST", "").strip()
    base_url = os.getenv("LANGFUSE_BASE_URL", "").strip()
    endpoint = host or base_url

    if not pub or not sec or not endpoint:
        return None

    # Build kwargs compatible across SDK versions
    sig = inspect.signature(Langfuse)
    params = sig.parameters

    kwargs = {"public_key": pub, "secret_key": sec}

    if "host" in params:
        kwargs["host"] = endpoint
    elif "base_url" in params:
        kwargs["base_url"] = endpoint

    return Langfuse(**kwargs)