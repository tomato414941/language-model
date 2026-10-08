import json
import threading
import unittest
import urllib.error
import urllib.request

from server import create_server


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, payload):
        request = urllib.request.Request(
            self.origin + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            return json.load(response)

    def test_reports_health_and_loaded_model(self):
        """ヘルスチェックと読み込んだモデルの情報を返す。"""
        with urllib.request.urlopen(self.origin + "/healthz") as response:
            self.assertEqual(json.load(response), {"status": "ok"})
        with urllib.request.urlopen(self.origin + "/") as response:
            metadata = json.load(response)
        self.assertEqual(metadata["parameters"], 4192)
        self.assertEqual(metadata["context_length"], 16)

    def test_generates_names_with_prefix_and_reproducible_seed(self):
        """指定された接頭辞・件数・シードで再現可能な名前を生成する。"""
        payload = {"prefix": "ka", "count": 3, "seed": 42, "max_new_tokens": 4}
        first = self.post(payload)
        self.assertEqual(first, self.post(payload))
        self.assertEqual(len(first["names"]), 3)
        self.assertTrue(all(name.startswith("ka") and len(name) <= 6 for name in first["names"]))

    def test_rejects_invalid_generation_options_and_keeps_serving(self):
        """不正な生成条件に400を返し、続く正常な生成を受け付ける。"""
        for payload in [{"prefix": "あ"}, {"count": 100}, {"temperature": -1}]:
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.post(payload)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        self.assertEqual(len(self.post({"count": 1})["names"]), 1)


if __name__ == "__main__":
    unittest.main()
