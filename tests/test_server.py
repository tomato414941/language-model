import unittest

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

    def test_reports_health(self):
        """ヘルスチェックに応答する。"""
        health = self.client.get("/healthz")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})

    def test_publishes_response_contract_and_interactive_docs(self):
        """Responses APIの仕様と対話式のAPI仕様を公開する。"""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/v1/responses", response.json()["paths"])
        docs = self.client.get("/docs")
        self.assertEqual(docs.status_code, 200)
        self.assertIn("text/html", docs.headers["Content-Type"])


if __name__ == "__main__":
    unittest.main()
