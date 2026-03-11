from __future__ import annotations

import time
from typing import Any, Callable, Optional
from google.genai.errors import ClientError


def gemini_call_safe(fn: Callable[[], Any], *, max_retries: int = 1) -> tuple[bool, Any, str]:
    """
    Wrap Gemini calls so 429/quota never breaks the API.

    Returns:
    (ok, result, error_type)
    """
    try:
        return True, fn(), ""
    except ClientError as e:
        # Quota / rate limit
        if getattr(e, "status_code", None) == 429:
            # optional single retry using the retryDelay info if present (best-effort)
            if max_retries > 0:
                try:
                    # e.message sometimes includes "Please retry in Xs"
                    msg = str(e)
                    # quick parse fallback
                    if "retryDelay" in msg or "retry in" in msg.lower():
                        time.sleep(2)  # keep short; don't block too long
                    return gemini_call_safe(fn, max_retries=max_retries - 1)
                except Exception:
                    pass
            return False, None, "quota_exceeded"
        return False, None, "client_error"
    except Exception:
        return False, None, "unknown_error"
