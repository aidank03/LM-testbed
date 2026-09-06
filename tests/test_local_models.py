import io
import json
import unittest
from unittest.mock import patch

from factor.local_models import LocalChatClient, LocalModelError, MAX_RESPONSE_BYTES, _NoRedirect


class LocalModelChecks(unittest.TestCase):
    def test_training_split_family_isolation(self):
        from experiments.local_model.dataset import build, LABELS
        partitions = build()
        seen_families, seen_text = set(), set()
        for rows in partitions.values():
            families = {r["family"] for r in rows}
            text = {r["text"] for r in rows}
            self.assertFalse(families & seen_families)
            self.assertFalse(text & seen_text)
            for family in families:
                self.assertEqual({r["label"] for r in rows if r["family"] == family}, set(LABELS))
            seen_families.update(families)
            seen_text.update(text)

    def test_remote_destination_rejected(self):
        for endpoint in ["https://example.com/v1", "http://127.0.0.1.evil/v1",
                         "http://user:secret@localhost/v1"]:
            with self.assertRaises(ValueError):
                LocalChatClient(endpoint)

    def test_request_and_trace(self):
        response = {"model": "fixture", "choices": [{"message": {"content": '{"tool":"finish"}'},
                     "finish_reason": "stop"}], "usage": {"completion_tokens": 8}}
        with patch("factor.local_models.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as call:
            result = LocalChatClient().complete([{"role": "user", "content": "Fixture"}],
                       response_format={"type": "json_object"})
        request = json.loads(call.call_args.args[0].data)
        self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(result["raw_response"], response)
        self.assertEqual(result["finish_reason"], "stop")

    def test_malformed_response_fails_closed(self):
        with patch("factor.local_models.urlopen", return_value=io.BytesIO(b'{"choices":[]}')):
            with self.assertRaises(LocalModelError):
                LocalChatClient().complete([{"role": "user", "content": "Fixture"}])

    def test_native_transport_disables_reasoning_and_redacts(self):
        raw = {"output": [{"type": "message", "content": '{"ok":true}'},
                          {"type": "reasoning", "content": "private fixture"}],
               "stats": {"total_output_tokens": 6}}
        with patch("factor.local_models.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())) as call:
            result = LocalChatClient(transport="lmstudio-native").complete(
                [{"role": "user", "content": "Fixture"}], timeout=3)
        request = json.loads(call.call_args.args[0].data)
        self.assertEqual(request["reasoning"], "off")
        self.assertFalse(request["store"])
        self.assertEqual(call.call_args.kwargs["timeout"], 3)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertNotIn("private fixture", json.dumps(result))

    def test_redirects_are_rejected(self):
        with self.assertRaises(LocalModelError):
            _NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://external.example")

    def test_oversized_response_is_rejected(self):
        with patch("factor.local_models.urlopen", return_value=io.BytesIO(b" "*(MAX_RESPONSE_BYTES+1))):
            with self.assertRaises(LocalModelError):
                LocalChatClient().complete([{"role": "user", "content": "Fixture"}])

    def test_zero_timeout_is_rejected(self):
        with self.assertRaises(ValueError):
            LocalChatClient().complete([{"role": "user", "content": "Fixture"}], timeout=0)


if __name__ == "__main__":
    unittest.main()
