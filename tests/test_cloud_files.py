import hashlib
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from scripts.cloud_files import transfer_file


class CloudFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.objects = {"/source": b"bilingual training bytes"}
        self.requests = []
        self.ranges = []
        self.reports_etag = False
        self.serves_ranges = False
        self.stop_after = None
        objects, requests, tests = self.objects, self.requests, self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(("GET", self.path))
                data = objects[self.path]
                requested = self.headers.get("Range")
                if requested and tests.serves_ranges:
                    first, last = requested.removeprefix("bytes=").split("-")
                    start, end = int(first), int(last) + 1
                    tests.ranges.append((start, end))
                    body = data[start:end]
                    self.send_response(206)
                    self.send_header(
                        "Content-Range", f"bytes {start}-{end - 1}/{len(data)}"
                    )
                else:
                    body = data
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if tests.stop_after is not None:
                    body, tests.stop_after = body[: tests.stop_after], None
                self.wfile.write(body)

            def do_PUT(self):
                requests.append(("PUT", self.path))
                objects[self.path] = self.rfile.read(
                    int(self.headers["Content-Length"])
                )
                self.send_response(200)
                if tests.reports_etag:
                    etag = hashlib.md5(objects[self.path]).hexdigest()
                    self.send_header("ETag", f'"{etag}"')
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def download_entry(self):
        data = self.objects["/source"]
        return {
            "path": "data/train.bin",
            "url": self.url + "/source",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def test_download_verifies_contents_and_reuses_the_verified_file(self):
        entry = self.download_entry()
        first = transfer_file("download", entry, self.root, attempts=1)
        second = transfer_file("download", entry, self.root, attempts=1)
        self.assertEqual(
            (self.root / entry["path"]).read_bytes(), self.objects["/source"]
        )
        self.assertTrue(first["verified"])
        self.assertTrue(second["reused"])
        self.assertEqual(self.requests, [("GET", "/source")])

    def test_download_preserves_existing_contents_when_received_checksum_is_invalid(
        self,
    ):
        entry = self.download_entry()
        path = self.root / entry["path"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"previous model data")
        self.objects["/source"] = b"corrupt incoming bytes"
        with self.assertRaisesRegex(ValueError, "expected checksum"):
            transfer_file("download", entry, self.root, attempts=1)
        self.assertEqual(path.read_bytes(), b"previous model data")

    def test_download_continues_a_response_that_stops_early(self):
        self.serves_ranges = True
        self.stop_after = 10
        entry = self.download_entry()
        transfer_file("download", entry, self.root, attempts=1)
        self.assertEqual(
            (self.root / entry["path"]).read_bytes(), self.objects["/source"]
        )
        self.assertEqual(self.ranges, [(0, 24), (10, 24)])

    def test_download_assembles_parts_fetched_over_parallel_connections(self):
        self.serves_ranges = True
        entry = self.download_entry()
        with mock.patch("scripts.cloud_files.PART_BYTES", 5):
            transfer_file("download", entry, self.root, attempts=1)
        self.assertEqual(
            (self.root / entry["path"]).read_bytes(), self.objects["/source"]
        )
        self.assertEqual(
            sorted(self.ranges), [(0, 5), (5, 10), (10, 15), (15, 20), (20, 24)]
        )

    def test_download_refuses_a_whole_object_sent_for_one_part(self):
        entry = self.download_entry()
        with (
            mock.patch("scripts.cloud_files.PART_BYTES", 5),
            self.assertRaisesRegex(ValueError, "ignored a range request"),
        ):
            transfer_file("download", entry, self.root, attempts=1)
        self.assertFalse((self.root / entry["path"]).exists())

    def test_upload_verifies_the_remote_contents_before_reporting_success(self):
        path = self.root / "model.pt"
        path.write_bytes(b"saved model weights")
        entry = {
            "path": path.name,
            "url": self.url + "/model",
            "verify_url": self.url + "/model",
        }
        result = transfer_file("upload", entry, self.root, attempts=1)
        self.assertEqual(self.objects["/model"], path.read_bytes())
        self.assertTrue(result["verified"])
        self.assertEqual(
            result["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
        )
        self.assertEqual(self.requests, [("PUT", "/model"), ("GET", "/model")])

    def test_upload_accepts_a_matching_etag_without_reading_back(self):
        self.reports_etag = True
        path = self.root / "model.pt"
        path.write_bytes(b"saved model weights")
        entry = {
            "path": path.name,
            "url": self.url + "/model",
            "verify_url": self.url + "/source",
        }
        result = transfer_file("upload", entry, self.root, attempts=1)
        self.assertEqual(self.objects["/model"], path.read_bytes())
        self.assertTrue(result["verified"])
        self.assertEqual(self.requests, [("PUT", "/model")])

    def test_upload_reports_when_readback_differs_from_the_source(self):
        path = self.root / "model.pt"
        path.write_bytes(b"saved model weights")
        entry = {
            "path": path.name,
            "url": self.url + "/model",
            "verify_url": self.url + "/source",
        }
        with self.assertRaisesRegex(ValueError, "source checksum"):
            transfer_file("upload", entry, self.root, attempts=1)


if __name__ == "__main__":
    unittest.main()
