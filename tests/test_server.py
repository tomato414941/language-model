import unittest

from fastapi.testclient import TestClient

from responses_api import ModelSettings
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

    def test_serves_a_chat_page_for_the_served_model(self):
        """サーバーのモデルと出力の上限で Responses API と話すページを、キーなしで配る。"""
        settings = ModelSettings(model_id="test-model", max_output_tokens=96, api_key="key")
        page = TestClient(create_app(response_settings=settings)).get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["Content-Type"])
        self.assertIn('data-model="test-model"', page.text)
        self.assertIn('data-max-output-tokens="96"', page.text)
        self.assertIn('fetch("/v1/responses"', page.text)


if __name__ == "__main__":
    unittest.main()
