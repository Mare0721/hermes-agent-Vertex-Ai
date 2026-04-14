"""Tests for Vertex AI adapter payload mappings."""

import agent.models.vertex_ai as vertex_ai_module


def test_build_vertex_payload_includes_google_search_from_extra_body():
    payload = vertex_ai_module._build_vertex_payload(
        [{"role": "user", "content": "Find recent updates."}],
        {"extra_body": {"googleSearch": {}}},
    )

    assert payload["tools"] == [{"googleSearch": {}}]


def test_build_vertex_payload_includes_google_search_retrieval_from_extra_body():
    retrieval = {
        "dynamicRetrievalConfig": {
            "mode": "MODE_DYNAMIC",
            "dynamicThreshold": 0.25,
        }
    }
    payload = vertex_ai_module._build_vertex_payload(
        [{"role": "user", "content": "Find docs."}],
        {"extra_body": {"googleSearchRetrieval": retrieval}},
    )

    assert payload["tools"] == [{"googleSearchRetrieval": retrieval}]


def test_build_vertex_payload_appends_google_search_to_function_tools():
    payload = vertex_ai_module._build_vertex_payload(
        [{"role": "user", "content": "Search and then compute."}],
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "extra_body": {"googleSearch": {}},
        },
    )

    assert len(payload["tools"]) == 2
    assert payload["tools"][0]["functionDeclarations"][0]["name"] == "web_search"
    assert payload["tools"][1] == {"googleSearch": {}}
