"""Optional sign-in: the user ids that are allowed to sign in.

There is no password. A user id is accepted only if it is listed in `<USERS_PATH>/users.txt` (one id per
line, `#` starts a comment), so that arbitrary names never end up on traces. This identifies a user, it
does not authenticate one: anyone who knows a listed id can use it.

The file is re-read when its modification time changes, so edits apply without restarting the app
(on Kubernetes the file comes from the `travel-advisor-users` ConfigMap).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

ANONYMOUS_USER_ID = "anonymous"
USERS_FILENAME = "users.txt"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._@-]{1,64}$")


class UserDirectory:
    """The allowed user ids, matched case-insensitively and returned as written in the file."""

    def __init__(self, path: str):
        self._file = os.path.join(path, USERS_FILENAME)
        self._lock = threading.Lock()
        self._signature = None
        self._users: Dict[str, str] = {}

    def _reload_if_changed(self) -> None:
        try:
            stat = os.stat(self._file)
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            signature = None
        if signature == self._signature:
            return
        with self._lock:
            if signature == self._signature:
                return
            users: Dict[str, str] = {}
            if signature is None:
                logger.warning(f"User list not found at {self._file}: nobody can sign in")
            else:
                with open(self._file, encoding="utf-8") as handle:
                    for raw in handle:
                        line = raw.split("#", 1)[0].strip()
                        if not line:
                            continue
                        if not _ID_PATTERN.match(line) or line.lower() == ANONYMOUS_USER_ID:
                            logger.warning(f"Ignoring invalid user id in {self._file}: {line[:64]!r}")
                            continue
                        users.setdefault(line.lower(), line)
                logger.info(f"Loaded {len(users)} user ids from {self._file}")
            self._users = users
            self._signature = signature

    def resolve(self, user_id: Optional[str]) -> Optional[str]:
        """The listed id (with the casing used in the file) for `user_id`, or None if it is not listed."""
        candidate = (user_id or "").strip()
        if not _ID_PATTERN.match(candidate):
            return None
        self._reload_if_changed()
        return self._users.get(candidate.lower())


def resolve_request_user(directory: UserDirectory, user_id: Optional[str]) -> str:
    """The user id to record for a request: a listed id, otherwise anonymous."""
    if not user_id:
        return ANONYMOUS_USER_ID
    resolved = directory.resolve(user_id)
    if resolved is None:
        logger.warning(f"Unknown user id on request, using {ANONYMOUS_USER_ID}: {user_id[:64]!r}")
        return ANONYMOUS_USER_ID
    return resolved
