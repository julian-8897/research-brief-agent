from docker.smoke_openai import completion_for


def test_smoke_provider_drives_search_fulltext_then_final():
    first = completion_for({"model": "smoke", "messages": []})
    first_message = first["choices"][0]["message"]
    assert first_message["tool_calls"][0]["function"]["name"] == "search_papers"

    search_result = {
        "role": "tool",
        "tool_call_id": "smoke-search",
        "content": '{"papers":[{"id":"2401.00001","title":"Paper"}]}',
    }
    second = completion_for({"model": "smoke", "messages": [search_result]})
    second_message = second["choices"][0]["message"]
    assert second_message["tool_calls"][0]["function"]["name"] == "get_full_text"

    fulltext_result = {
        "role": "tool",
        "tool_call_id": "smoke-fulltext",
        "content": '{"papers":[{"id":"2401.00001","full_text":"body"}]}',
    }
    final = completion_for(
        {"model": "smoke", "messages": [search_result, fulltext_result]}
    )
    final_message = final["choices"][0]["message"]
    assert "tool_calls" not in final_message
    assert "# Decision Memo" in final_message["content"]
    assert "[2401.00001]" in final_message["content"]
