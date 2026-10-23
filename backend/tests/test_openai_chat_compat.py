from types import SimpleNamespace

import pytest

from app.utils.openai_chat_compat import (
    create_chat_completion,
    create_json_chat_completion,
    extract_chat_completion_text,
    is_gpt5_family,
    is_response_format_unsupported,
)


class CompletionRecorder:
    def __init__(self, result=None, error=None):
        self.result = result or object()
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def client_for(recorder):
    return SimpleNamespace(chat=SimpleNamespace(completions=recorder))


def test_gpt5_uses_completion_token_limit_without_temperature():
    recorder = CompletionRecorder()
    messages = [{"role": "user", "content": "hello"}]

    result = create_chat_completion(
        client_for(recorder),
        model="gpt-5-2025-08-07",
        messages=messages,
        temperature=0.2,
        max_tokens=123,
        response_format={"type": "json_object"},
    )

    assert result is recorder.result
    assert recorder.calls == [
        {
            "model": "gpt-5-2025-08-07",
            "messages": messages,
            "max_completion_tokens": 123,
            "response_format": {"type": "json_object"},
        }
    ]


def test_legacy_model_preserves_original_request_shape():
    recorder = CompletionRecorder()
    messages = [{"role": "user", "content": "hello"}]

    create_chat_completion(
        client_for(recorder),
        model="third-party-chat-model",
        messages=messages,
        temperature=0.7,
        max_tokens=456,
        response_format={"type": "json_object"},
    )

    assert recorder.calls == [
        {
            "model": "third-party-chat-model",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 456,
            "response_format": {"type": "json_object"},
        }
    ]


def test_provider_error_is_propagated_without_guessing_or_retrying():
    provider_error = RuntimeError("unsupported max_tokens due to a server outage")
    recorder = CompletionRecorder(error=provider_error)

    with pytest.raises(RuntimeError) as captured:
        create_chat_completion(
            client_for(recorder),
            model="legacy-model",
            messages=[],
            max_tokens=10,
        )

    assert captured.value is provider_error
    assert len(recorder.calls) == 1


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5", True),
        (" GPT-5.1-mini ", True),
        ("gpt-4.1", False),
        ("my-gpt-5-proxy", False),
        (None, False),
    ],
)
def test_gpt5_family_detection(model, expected):
    assert is_gpt5_family(model) is expected


def test_extracts_text_from_supported_content_shapes():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"text": {"value": "first"}},
                        {"content": " second"},
                        SimpleNamespace(text=" third"),
                    ]
                )
            )
        ]
    )

    assert extract_chat_completion_text(response) == "first second third"
    assert extract_chat_completion_text(SimpleNamespace(choices=[])) == ""


class ProviderError(RuntimeError):
    def __init__(self, *, status_code, body):
        super().__init__("provider error")
        self.status_code = status_code
        self.body = body


class SequenceRecorder:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


# The OpenAI SDK exposes the inner error object as `body`.
ANTHROPIC_JSON_MODE_REJECTION = {
    "code": "invalid_request_error",
    "message": "response_format.type: Input should be 'json_schema'",
    "type": "invalid_request_error",
    "param": None,
}


@pytest.mark.parametrize(
    ("status_code", "body", "expected"),
    [
        (400, ANTHROPIC_JSON_MODE_REJECTION, True),
        (
            400,
            {"error": {"param": "response_format", "message": "not supported"}},
            True,
        ),
        (
            400,
            {"error": {"message": "messages.0.content: Input should be a valid string"}},
            False,
        ),
        (500, ANTHROPIC_JSON_MODE_REJECTION, False),
    ],
)
def test_response_format_rejection_detection(status_code, body, expected):
    error = ProviderError(status_code=status_code, body=body)

    assert is_response_format_unsupported(error) is expected


def test_json_completion_retries_once_without_json_mode_after_rejection():
    recorder = SequenceRecorder(
        ProviderError(status_code=400, body=ANTHROPIC_JSON_MODE_REJECTION),
        "completion",
    )
    messages = [{"role": "user", "content": "Return JSON"}]

    result = create_json_chat_completion(
        client_for(recorder),
        model="claude-sonnet-4-6",
        messages=messages,
        temperature=0.7,
    )

    assert result == "completion"
    assert recorder.calls == [
        {
            "model": "claude-sonnet-4-6",
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
        },
        {"model": "claude-sonnet-4-6", "messages": messages, "temperature": 0.7},
    ]


def test_json_completion_propagates_unrelated_errors_without_retrying():
    provider_error = ProviderError(
        status_code=429, body={"error": {"message": "rate limited"}}
    )
    recorder = SequenceRecorder(provider_error, "unused")

    with pytest.raises(ProviderError) as captured:
        create_json_chat_completion(
            client_for(recorder), model="any-model", messages=[]
        )

    assert captured.value is provider_error
    assert len(recorder.calls) == 1
