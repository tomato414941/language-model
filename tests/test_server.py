import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import create_app


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def post(self, payload):
        response = self.client.post("/generate", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_reports_health_and_loaded_model(self):
        """ヘルスチェックと読み込んだモデルの情報を返す。"""
        health = self.client.get("/healthz")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["parameters"], 4192)
        self.assertEqual(response.json()["context_length"], 16)

    def test_generates_names_with_prefix_and_reproducible_seed(self):
        """指定された接頭辞・件数・シードで再現可能な名前を生成する。"""
        payload = {"prefix": "ka", "count": 3, "seed": 42}
        first = self.post(payload)
        self.assertEqual(first, self.post(payload))
        self.assertEqual(first["names"], ["karale", "kaboh", "kaliva"])
        limited = self.post({**payload, "max_new_tokens": 4})
        self.assertEqual(len(limited["names"]), 3)
        self.assertTrue(
            all(name.startswith("ka") and len(name) <= 6 for name in limited["names"])
        )
        self.assertEqual(
            len(self.post({"temperature": 0, "max_new_tokens": 1})["names"]), 1
        )

    def test_validates_generation_options_and_keeps_serving(self):
        """不正な生成条件に422を返し、続く正常な生成を受け付ける。"""
        for payload in [
            {"prefix": "あ"},
            {"prefix": "a" * 16},
            {"count": 100},
            {"count": True},
            {"temperature": -1},
            {"temperature": "0.8"},
            {"seed": -1},
            {"max_new_tokens": 17},
            {"unknown": True},
        ]:
            with self.subTest(payload=payload):
                response = self.client.post("/generate", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertTrue(response.json()["detail"])
        self.assertEqual(len(self.post({"count": 1})["names"]), 1)

    def test_returns_validation_errors_for_nonfinite_numbers_and_malformed_json(self):
        """有限でない数値や壊れたJSONに、JSON形式の検証エラーを返す。"""
        for body in [
            '{"temperature": NaN}',
            '{"temperature": Infinity}',
            '{"temperature": 1e309}',
            "{",
        ]:
            with self.subTest(body=body):
                response = self.client.post(
                    "/generate",
                    content=body,
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(response.status_code, 422, response.text)
                self.assertTrue(response.json()["detail"])

    def test_enforces_json_content_type_and_body_size(self):
        """JSONを4096バイトまで受け付け、形式やサイズの違反を通知する。"""
        response = self.client.post("/generate", content="{}")
        self.assertEqual(response.status_code, 415)
        headers = {"Content-Type": "application/json"}
        body = '{"seed": 0, "temperature": 0, "max_new_tokens": 1}'
        body += " " * (4096 - len(body))
        self.assertEqual(
            self.client.post("/generate", content=body, headers=headers).status_code,
            200,
        )
        response = self.client.post(
            "/generate", content=iter([body.encode(), b" "]), headers=headers
        )
        self.assertEqual(response.status_code, 413)
        self.assertTrue(response.json()["detail"])

    def test_keeps_health_available_when_generation_is_busy(self):
        """同時に2件を生成し、混雑時に再試行を案内しながらヘルスチェックに応答する。"""
        started, release = threading.Barrier(3), threading.Event()
        model = self.client.app.state.model
        original = model.generate

        def slow_generation(*args, **kwargs):
            started.wait(timeout=5)
            if not release.wait(timeout=5):
                raise TimeoutError("Generation was not released.")
            return original(*args, **kwargs)

        with (
            patch.object(model, "generate", slow_generation),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            futures = [
                pool.submit(self.client.post, "/generate", json={"max_new_tokens": 1})
                for _ in range(2)
            ]
            try:
                started.wait(timeout=5)
                response = self.client.post("/generate", json={})
                self.assertEqual(response.status_code, 429)
                self.assertEqual(response.headers["Retry-After"], "1")
                self.assertEqual(self.client.get("/healthz").json(), {"status": "ok"})
            finally:
                release.set()
            for future in futures:
                self.assertEqual(future.result(timeout=5).status_code, 200)
        self.assertEqual(len(self.post({"count": 1})["names"]), 1)

    def test_publishes_generation_contract_and_interactive_docs(self):
        """生成APIの入力範囲・応答形式と対話式のAPI仕様を公開する。"""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        specification = response.json()
        operation = specification["paths"]["/generate"]["post"]
        request_ref = operation["requestBody"]["content"]["application/json"]["schema"][
            "$ref"
        ]
        fields = specification["components"]["schemas"][request_ref.rsplit("/", 1)[-1]][
            "properties"
        ]
        self.assertEqual(fields["count"]["minimum"], 1)
        self.assertEqual(fields["count"]["maximum"], 20)
        response_ref = operation["responses"]["200"]["content"]["application/json"][
            "schema"
        ]["$ref"]
        output = specification["components"]["schemas"][response_ref.rsplit("/", 1)[-1]]
        self.assertIn("names", output["required"])
        self.assertIn("seed", output["required"])
        docs = self.client.get("/docs")
        self.assertEqual(docs.status_code, 200)
        self.assertIn("text/html", docs.headers["Content-Type"])


if __name__ == "__main__":
    unittest.main()
