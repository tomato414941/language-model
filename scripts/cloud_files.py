import argparse
import hashlib
import json
import os
import time
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_digest(response):
    checksum = hashlib.sha256()
    size = 0
    while chunk := response.read(1024 * 1024):
        checksum.update(chunk)
        size += len(chunk)
    return checksum.hexdigest(), size


def transfer_file(action, entry, root, attempts=3):
    root = Path(root).resolve()
    relative = Path(entry["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("transfer paths must stay within the project")
    if action == "download":
        expected = entry["sha256"]
        size = entry["bytes"]
        if path.is_file() and path.stat().st_size == size and digest(path) == expected:
            return {
                "path": str(relative),
                "sha256": expected,
                "bytes": size,
                "reused": True,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
    elif action == "upload":
        if not path.exists() and entry.get("optional", False):
            return {"path": str(relative), "skipped": True}
        expected, size = digest(path), path.stat().st_size
    else:
        raise ValueError("choose download or upload")
    temporary = path.with_name(path.name + ".download")
    for attempt in range(attempts):
        try:
            if action == "download":
                checksum = hashlib.sha256()
                received = 0
                with (
                    urlopen(entry["url"], timeout=120) as response,
                    temporary.open("wb") as target,
                ):
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
                        checksum.update(chunk)
                        received += len(chunk)
                if received != size or checksum.hexdigest() != expected:
                    raise ValueError(
                        "downloaded contents do not match the expected checksum"
                    )
                os.replace(temporary, path)
            else:
                with path.open("rb") as source:
                    request = Request(
                        entry["url"],
                        data=source,
                        method="PUT",
                        headers={"Content-Length": str(size)},
                    )
                    with urlopen(request, timeout=120) as response:
                        response.read()
                with urlopen(entry["verify_url"], timeout=120) as response:
                    checksum, received = read_digest(response)
                if received != size or checksum != expected:
                    raise ValueError(
                        "uploaded contents do not match the source checksum"
                    )
            return {
                "path": str(relative),
                "sha256": expected,
                "bytes": size,
                "verified": True,
            }
        except (OSError, ValueError, HTTPException):
            if action == "download":
                temporary.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Transfer private training files with checksum verification"
    )
    parser.add_argument("action", choices=("download", "upload"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    entries = json.loads(args.manifest.read_text())[args.action + "s"]
    receipt = {"action": args.action, "status": "running", "files": []}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(temporary, args.receipt)

    save()
    for entry in entries:
        print(json.dumps({"action": args.action, "path": entry["path"]}), flush=True)
        try:
            receipt["files"].append(transfer_file(args.action, entry, args.root))
        except (OSError, ValueError, HTTPException, KeyError, TypeError) as error:
            receipt.update(
                status="failed",
                failed_path=entry["path"],
                error_type=type(error).__name__,
            )
            save()
            parser.exit(
                1,
                f"{args.action} failed for {entry['path']} ({type(error).__name__})\n",
            )
        save()
    receipt["status"] = "verified"
    save()
    print(json.dumps({"action": args.action, "status": "verified"}), flush=True)


if __name__ == "__main__":
    main()
