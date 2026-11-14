"""Optional handoff to the Job Manager plugin.

If the user has Job Manager installed, the input we just wrote can be sent
straight to a cluster. If they do not, nothing about this plugin should change:
the button is not shown at all, no import is attempted, no error is logged.

Job Manager is found through the host's plugin list rather than imported, since
plugins are not importable by name from each other, and it is identified by the
public ``submit_file`` entry point it documents -- a plugin that merely shares
the name but predates that API is correctly treated as absent.
"""

import logging
import os

JOB_MANAGER_PLUGIN_NAME = "Job Manager"


def find_job_manager(main_window):
    """Return the Job Manager module, or None when it is not available."""
    manager = getattr(main_window, "plugin_manager", None)
    for record in getattr(manager, "plugins", None) or []:
        try:
            if record.get("name") != JOB_MANAGER_PLUGIN_NAME:
                continue
            module = record.get("module")
        except AttributeError:
            continue
        if module is not None and callable(getattr(module, "submit_file", None)):
            return module
    return None


def is_available(main_window) -> bool:
    return find_job_manager(main_window) is not None


def submit_to_cluster(main_window, paths, name="") -> bool:
    """Hand ``paths`` (one path or a list, input first) to Job Manager's submit wizard.

    False if Job Manager is absent, a file is missing, or it declined.
    """
    if isinstance(paths, str):
        paths = [paths]
    paths = [p for p in (paths or []) if p]
    module = find_job_manager(main_window)
    if module is None or not paths or not all(os.path.isfile(p) for p in paths):
        return False
    try:
        return bool(module.submit_file(paths if len(paths) > 1 else paths[0], name=name or ""))
    except Exception:
        logging.warning("Could not hand the input to Job Manager", exc_info=True)
        return False
