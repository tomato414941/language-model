import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen

# A download arrives in parts over several connections, since one connection
# to a distant store can be slow, and a connection that closes early costs
# only the rest of its part.
PART_BYTES = 64 * 1024 * 1024
CONNECTIONS = 8


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def upload_digests(path):
    """Return the SHA-256 and MD5 of a file, read once."""
    checksum, content = hashlib.sha256(), hashlib.md5()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            checksum.update(chunk)
            content.update(chunk)
    return checksum.hexdigest(), content.hexdigest()


def read_digest(response):
    checksum = hashlib.sha256()
    size = 0
    while chunk := response.read(1024 * 1024):
        checksum.update(chunk)
        size += len(chunk)
    return checksum.hexdigest(), size


def fetch_part(url, descriptor, start, end, size, attempts):
    """Write bytes start to end - 1 of the object at the same file offsets.

    A response that stops early ends without an error, so the part continues
    from where its data stopped; only requests that bring nothing count
    against the attempts.
    """
    position, stalled, error = start, 0, None
    while position < end:
        before = position
        try:
            request = Request(url, headers={"Range": f"bytes={position}-{end - 1}"})
            with urlopen(request, timeout=120) as response:
                if response.status == 206:
                    total = response.headers.get("Content-Range", "").rpartition("/")[2]
                elif (position, end) == (0, size):
                    # A store may answer a range covering everything in full.
                    total = response.headers.get("Content-Length")
                else:
                    raise ValueError("the store ignored a range request")
                if total != str(size):
                    raise ValueError(
                        "downloaded contents do not match the expected checksum"
                    )
                while position < end and (
                    chunk := response.read(min(1024 * 1024, end - position))
                ):
                    os.pwrite(descriptor, chunk, position)
                    position += len(chunk)
        except (OSError, HTTPException) as failure:
            error = failure
        if position == before:
            stalled += 1
            if stalled == attempts:
                raise error or ValueError("downloaded contents ended early")
            time.sleep(2**stalled)


def download(url, temporary, size, attempts):
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(descriptor, size)
        pool = ThreadPoolExecutor(CONNECTIONS)
        try:
            parts = [
                pool.submit(
                    fetch_part, url, descriptor, start, min(start + PART_BYTES, size), size, attempts
                )
                for start in range(0, size, PART_BYTES)
            ]
            for part in parts:
                part.result()
        finally:
            pool.shutdown(cancel_futures=True)
    finally:
        os.close(descriptor)


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
        (expected, content_md5), size = upload_digests(path), path.stat().st_size
    else:
        raise ValueError("choose download or upload")
    temporary = path.with_name(path.name + ".download")
    for attempt in range(attempts):
        try:
            if action == "download":
                download(entry["url"], temporary, size, attempts)
                if digest(temporary) != expected:
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
                        etag = (response.headers.get("ETag") or "").strip('"')
                # A store that reports the MD5 of what it received has already
                # confirmed the contents; read back only when it does not.
                if etag != content_md5:
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
