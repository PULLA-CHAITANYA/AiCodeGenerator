import unittest

import app as app_module


class GenerateRouteTests(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()
        self.original_api_key = app_module.TOGETHER_API_KEY

    def tearDown(self):
        app_module.TOGETHER_API_KEY = self.original_api_key

    def test_capability_question_returns_local_response(self):
        app_module.TOGETHER_API_KEY = None

        response = self.client.post(
            "/generate",
            json={
                "prompt": "oh cool an agent here, what can u do ?",
                "language": "Python",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["response_type"], "capabilities")
        self.assertTrue(response.get_json()["assistant_message"].strip())

    def test_problem_generation_still_requires_api_key(self):
        app_module.TOGETHER_API_KEY = None

        response = self.client.post(
            "/generate",
            json={
                "prompt": "Write a factorial function",
                "language": "Python",
            },
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn("API key is not configured", response.get_json()["error"])

    def test_capability_matcher_accepts_supported_casual_question(self):
        self.assertTrue(app_module.is_capability_question("oh cool an agent here, what can u do ?"))

    def test_capability_matcher_rejects_embedded_code_request(self):
        self.assertFalse(app_module.is_capability_question("write a what can you do function"))


if __name__ == "__main__":
    unittest.main()
