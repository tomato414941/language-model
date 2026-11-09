"""What a step owes a run carried on rented machines by job-runner: its progress, the files to keep, whole outputs.

The run gives a step two places in its environment: JOB_RUNNER_PROGRESS, a file for
how far it has got, and JOB_RUNNER_KEEP, a directory where a file it wants kept, such
as a checkpoint just saved, is linked under its path, with each "/" as "%2F".
Elsewhere neither is set and both calls do nothing. Run again, a step continues
from what it kept, and its outputs appear only whole.
"""

import json
import os
from pathlib import Path
from urllib.parse import quote


def progress(done, total):
    """Report how far the step has got, in the units its expected rate is given in."""
    path = os.environ.get("JOB_RUNNER_PROGRESS")
    if path:
        temporary = Path(f"{path}.{os.getpid()}")
        temporary.write_text(json.dumps({"done": done, "total": total}))
        os.replace(temporary, path)


def keep(path):
    """Ask the run to keep a file just saved; a newer one at the same path takes its place."""
    directory = os.environ.get("JOB_RUNNER_KEEP")
    if not directory:
        return
    relative = os.path.relpath(Path(path).resolve(), Path.cwd().resolve())
    if relative.startswith(".."):
        raise ValueError("a step keeps only files within its working directory")
    name = quote(relative, safe="")
    temporary = Path(directory) / f".{name}.{os.getpid()}"
    os.link(path, temporary)
    os.replace(temporary, Path(directory) / name)


def write_output(path, text):
    """Write an output whole: it appears only once complete, so that a step run again can trust it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
