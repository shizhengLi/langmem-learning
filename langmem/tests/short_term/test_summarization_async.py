import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately

from langmem.short_term.summarization import asummarize_messages, SummarizationNode
from tests.short_term.utils import FakeChatModel

pytestmark = pytest.mark.anyio

async def test_empty_input():
    model = FakeChatModel(responses=[])

    # Test with empty message list
    result = await asummarize_messages(
        [],
        running_summary=None,
        model=model,
        max_tokens=10,
        max_summary_tokens=1,
    )

    # Check that no summarization occurred
    assert result.running_summary is None
    assert result.messages == []
    assert len(model.invoke_calls) == 0

    # Test with only system message
    system_msg = SystemMessage(content="You are a helpful assistant.", id="sys")
    result = await asummarize_messages(
        [system_msg],
        running_summary=None,
        model=model,
        max_tokens=10,
        max_summary_tokens=1,
    )

    # Check that no summarization occurred
    assert result.running_summary is None
    assert result.messages == [system_msg]


async def test_no_summarization_needed():
    model = FakeChatModel(responses=[])

    messages = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
    ]

    # Tokens are under the limit, so no summarization should occur
    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=10,
        max_summary_tokens=1,
    )

    # Check that no summarization occurred
    assert result.running_summary is None
    assert result.messages == messages
    assert len(model.invoke_calls) == 0  # Model should not have been called


async def test_summarize_first_time():
    model = FakeChatModel(
        responses=[AIMessage(content="This is a summary of the conversation.")]
    )

    # Create enough messages to trigger summarization
    messages = [
        # these messages will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        # these messages will be added to the result post-summarization
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        HumanMessage(content="Latest message", id="9"),
    ]

    # Call the summarizer
    max_summary_tokens = 1
    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=max_summary_tokens,
    )

    # Check that model was called
    assert len(model.invoke_calls) == 1

    # Check that the result has the expected structure:
    # - First message should be a summary
    # - Last 3 messages should be the last 3 original messages
    assert len(result.messages) == 4
    assert result.messages[0].type == "system"
    assert "summary" in result.messages[0].content.lower()
    assert result.messages[1:] == messages[-3:]

    # Check the summary value
    summary_value = result.running_summary
    assert summary_value is not None
    assert summary_value.summary == "This is a summary of the conversation."
    assert summary_value.summarized_message_ids == set(
        msg.id for msg in messages[:6]
    )

    # Test subsequent invocation (no new summary needed)
    result = await asummarize_messages(
        messages,
        running_summary=summary_value,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=max_summary_tokens,
    )
    assert len(result.messages) == 4
    assert result.messages[0].type == "system"
    assert (
        result.messages[0].content
        == "Summary of the conversation so far: This is a summary of the conversation."
    )
    assert result.messages[1:] == messages[-3:]


async def test_max_tokens_before_summary():
    model = FakeChatModel(
        responses=[AIMessage(content="This is a summary of the conversation.")]
    )

    # Create enough messages to trigger summarization
    messages = [
        # these messages will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        # these messages will be added to the result post-summarization
        HumanMessage(content="Latest message", id="9"),
    ]

    # Call the summarizer
    max_summary_tokens = 1
    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_tokens_before_summary=8,
        max_summary_tokens=max_summary_tokens,
    )

    # Check that model was called
    assert len(model.invoke_calls) == 1

    # Check that the result has the expected structure:
    # - First message should be a summary
    # - Last message should be the last original message
    assert len(result.messages) == 2
    assert result.messages[0].type == "system"
    assert "summary" in result.messages[0].content.lower()
    assert result.messages[1:] == messages[-1:]

    # Check the summary value
    summary_value = result.running_summary
    assert summary_value is not None
    assert summary_value.summary == "This is a summary of the conversation."
    assert summary_value.summarized_message_ids == set(
        msg.id for msg in messages[:8]
    )  # All messages except the latest

    # Test subsequent invocation (no new summary needed)
    result = await asummarize_messages(
        messages,
        running_summary=summary_value,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_tokens_before_summary=8,
        max_summary_tokens=max_summary_tokens,
    )
    assert len(result.messages) == 2
    assert result.messages[0].type == "system"
    assert (
        result.messages[0].content
        == "Summary of the conversation so far: This is a summary of the conversation."
    )
    assert result.messages[1:] == messages[-1:]


async def test_with_system_message():
    """Test summarization with a system message present."""
    model = FakeChatModel(
        responses=[AIMessage(content="Summary with system message present.")]
    )

    # Create messages with a system message
    messages = [
        # this will not be summarized, but will be added post-summarization
        SystemMessage(content="You are a helpful assistant.", id="0"),
        # these messages will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        # these messages will be added to the result post-summarization
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        HumanMessage(content="Latest message", id="9"),
    ]

    # Call the summarizer
    max_tokens = 6
    # we're using len() as a token counter, so a summary is simply 1 "token"
    max_summary_tokens = 1
    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
    )

    # Check that model was called
    assert len(model.invoke_calls) == 1
    assert model.invoke_calls[0] == messages[1:7] + [
        HumanMessage(content="Create a summary of the conversation above:")
    ]

    # Check that the result has the expected structure:
    # - System message should be preserved
    # - Second message should be a summary of messages 2-5
    # - Last 3 messages should be the last 3 original messages
    assert len(result.messages) == 5
    assert result.messages[0].type == "system"
    assert result.messages[1].type == "system"  # Summary message
    assert "summary" in result.messages[1].content.lower()
    assert result.messages[2:] == messages[-3:]


async def test_approximate_token_counter():
    model = FakeChatModel(responses=[AIMessage(content="Summary with empty messages.")])

    # Create messages with some empty content
    messages = [
        # these will be summarized
        HumanMessage(content="", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        # these will be added to the result post-summarization
        HumanMessage(content="Latest message", id="9"),
    ]

    # Call the summarizer
    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=count_tokens_approximately,
        max_tokens=50,
        max_summary_tokens=10,
    )

    # Check that summarization still works with empty messages
    assert len(result.messages) == 2
    assert "summary" in result.messages[0].content.lower()
    assert result.messages[1:] == messages[-1:]


async def test_large_number_of_messages():
    """Test summarization with a large number of messages."""
    model = FakeChatModel(responses=[AIMessage(content="Summary of many messages.")])

    # Create a large number of messages
    messages = []
    for i in range(20):  # 20 pairs of messages = 40 messages total
        messages.append(HumanMessage(content=f"Human message {i}", id=f"h{i}"))
        messages.append(AIMessage(content=f"AI response {i}", id=f"a{i}"))

    # Add one final message
    messages.append(HumanMessage(content="Final message", id=f"h{len(messages)}"))

    # Call the summarizer
    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=22,
        max_summary_tokens=0,
    )

    # Check that summarization works with many messages
    assert (
        len(result.messages) == 20
    )  # summary (for the first 22 messages) + 19 remaining original messages
    assert "summary" in result.messages[0].content.lower()
    assert result.messages[1:] == messages[22:]  # last 19 original messages

    # Check that the model was called with a subset of messages
    # The implementation might limit how many messages are sent to the model
    assert len(model.invoke_calls) == 1


async def test_subsequent_summarization_with_new_messages():
    model = FakeChatModel(
        responses=[
            AIMessage(content="First summary of the conversation."),
            AIMessage(content="Updated summary including new messages."),
        ]
    )

    # First batch of messages
    messages1 = [
        # these will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        # this will be propagated to the next summarization
        HumanMessage(content="Latest message 1", id="7"),  # will be filtered out
    ]

    # First summarization
    max_tokens = 6
    max_summary_tokens = 1
    result = await asummarize_messages(
        messages1,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
    )

    # Verify the first summarization result
    assert "summary" in result.messages[0].content.lower()
    assert len(result.messages) == 2
    assert result.messages[-1] == messages1[-1]
    assert len(model.invoke_calls) == 1

    # Check the summary value
    summary_value = result.running_summary
    assert summary_value.summary == "First summary of the conversation."
    assert len(summary_value.summarized_message_ids) == 6  # first 6 messages

    # Add more messages to trigger another summarization
    new_messages = [
        # these will be summarized (including accounting for the previous summary!)
        AIMessage(content="Response to latest 1", id="8"),  # will be filtered out
        HumanMessage(content="Message 4", id="9"),
        AIMessage(content="Response 4", id="10"),
        HumanMessage(content="Message 5", id="11"),
        AIMessage(content="Response 5", id="12"),
        # these will be kept in the final result
        HumanMessage(content="Message 6", id="13"),
        AIMessage(content="Response 6", id="14"),
        HumanMessage(content="Latest message 2", id="15"),
    ]

    messages2 = messages1.copy()
    messages2.extend(new_messages)

    # Second summarization
    result2 = await asummarize_messages(
        messages2,
        running_summary=summary_value,
        model=model,
        token_counter=len,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
    )

    # Check that model was called twice
    assert len(model.invoke_calls) == 2

    # Get the messages sent to the model in the second call
    second_call_messages = model.invoke_calls[1]

    # Check that the previous summary is included in the prompt
    prompt_message = second_call_messages[-1]
    assert "First summary of the conversation" in prompt_message.content
    assert "Extend this summary" in prompt_message.content

    # Check that only the new messages are sent to the model, not already summarized ones
    assert len(second_call_messages) == 5  # 4 messages + prompt
    assert [msg.content for msg in second_call_messages[:-1]] == [
        "Message 4",
        "Response 4",
        "Message 5",
        "Response 5",
    ]

    # Verify the structure of the final result
    assert "summary" in result2.messages[0].content.lower()
    assert len(result2.messages) == 4  # Summary + last 3 messages
    assert result2.messages[-3:] == messages2[-3:]

    # Check the updated summary
    updated_summary_value = result2.running_summary
    assert updated_summary_value.summary == "Updated summary including new messages."
    # Verify all messages except the last 3 were summarized
    assert len(updated_summary_value.summarized_message_ids) == len(messages2) - 3


async def test_subsequent_summarization_with_new_messages_approximate_token_counter():
    model = FakeChatModel(
        responses=[
            AIMessage(content="First summary of the conversation."),
            AIMessage(content="Updated summary including new messages."),
        ]
    )

    # First batch of messages
    messages1 = [
        # these will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        # this will be propagated to the next summarization
        HumanMessage(content="Latest message 1", id="7"),
    ]

    # First summarization
    max_tokens = 45
    max_summary_tokens = 15
    result = await asummarize_messages(
        messages1,
        running_summary=None,
        model=model,
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
    )

    # Verify the first summarization result
    assert "summary" in result.messages[0].content.lower()
    assert len(result.messages) == 2
    assert result.messages[-1] == messages1[-1]
    assert len(model.invoke_calls) == 1

    # Check the summary value
    summary_value = result.running_summary
    assert summary_value.summary == "First summary of the conversation."
    assert len(summary_value.summarized_message_ids) == 6  # first 6 messages

    # Add more messages to trigger another summarization
    new_messages = [
        # these will be summarized (including accounting for the previous summary!)
        AIMessage(content="Response to latest 1", id="8"),
        HumanMessage(content="Message 4", id="9"),
        AIMessage(content="Response 4", id="10"),
        HumanMessage(content="Message 5", id="11"),
        AIMessage(content="Response 5", id="12"),
        # these will be kept in the final result
        HumanMessage(content="Message 6", id="13"),
        AIMessage(content="Response 6", id="14"),
        HumanMessage(content="Latest message 2", id="15"),
    ]

    messages2 = messages1.copy()
    messages2.extend(new_messages)

    # Second summarization
    result2 = await asummarize_messages(
        messages2,
        running_summary=summary_value,
        model=model,
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
    )

    # Check that model was called twice
    assert len(model.invoke_calls) == 2

    # Get the messages sent to the model in the second call
    second_call_messages = model.invoke_calls[1]

    # Check that the previous summary is included in the prompt
    prompt_message = second_call_messages[-1]
    assert "First summary of the conversation" in prompt_message.content
    assert "Extend this summary" in prompt_message.content

    # Check that only the new messages are sent to the model, not already summarized ones
    assert len(second_call_messages) == 5  # 4 messages + prompt
    assert [msg.content for msg in second_call_messages[:-1]] == [
        "Message 4",
        "Response 4",
        "Message 5",
        "Response 5",
    ]

    # Verify the structure of the final result
    assert "summary" in result2.messages[0].content.lower()
    assert len(result2.messages) == 4  # Summary + last 3 messages
    assert result2.messages[-3:] == messages2[-3:]

    # Check the updated summary
    updated_summary_value = result2.running_summary
    assert updated_summary_value.summary == "Updated summary including new messages."
    # Verify all messages except the last 3 were summarized
    assert len(updated_summary_value.summarized_message_ids) == len(messages2) - 3


async def test_last_ai_with_tool_calls():
    model = FakeChatModel(responses=[AIMessage(content="Summary without tool calls.")])

    messages = [
        # these will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(
            content="",
            id="2",
            tool_calls=[
                {"name": "tool_1", "id": "1", "args": {"arg1": "value1"}},
                {"name": "tool_2", "id": "2", "args": {"arg1": "value1"}},
            ],
        ),
        ToolMessage(content="Call tool 1", tool_call_id="1", name="tool_1", id="3"),
        ToolMessage(content="Call tool 2", tool_call_id="2", name="tool_2", id="4"),
        # these will be kept in the final result
        AIMessage(content="Response 1", id="5"),
        HumanMessage(content="Message 2", id="6"),
    ]

    # Call the summarizer
    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens_before_summary=2,
        max_tokens=6,
        max_summary_tokens=1,
    )

    # Check that the AI message with tool calls was summarized together with the tool messages
    assert len(result.messages) == 3
    assert result.messages[0].type == "system"  # Summary
    assert result.messages[-2:] == messages[-2:]
    assert result.running_summary.summarized_message_ids == set(
        msg.id for msg in messages[:-2]
    )


async def test_missing_message_ids():
    messages = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response"),  # Missing ID
    ]
    with pytest.raises(ValueError, match="Messages are required to have ID field"):
        await asummarize_messages(
            messages,
            running_summary=None,
            model=FakeChatModel(responses=[]),
            max_tokens=10,
            max_summary_tokens=1,
        )


async def test_duplicate_message_ids():
    model = FakeChatModel(responses=[AIMessage(content="Summary")])

    # First summarization
    messages1 = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
    ]

    result = await asummarize_messages(
        messages1,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=2,
        max_summary_tokens=1,
    )

    # Second summarization with a duplicate ID
    messages2 = [
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="1"),  # Duplicate ID
    ]

    with pytest.raises(ValueError, match="has already been summarized"):
        await asummarize_messages(
            messages1 + messages2,
            running_summary=result.running_summary,
            model=model,
            token_counter=len,
            max_tokens=5,
            max_summary_tokens=1,
        )


async def test_summarization_updated_messages():
    # this is a variant of test_subsequent_summarization_with_new_messages
    # that passes the updated (ie., summarized) messages on the second turn
    model = FakeChatModel(
        responses=[
            AIMessage(content="First summary of the conversation."),
            AIMessage(content="Updated summary including new messages."),
        ]
    )

    # First batch of messages
    messages1 = [
        # these will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        # this will be propagated to the next summarization
        HumanMessage(content="Latest message 1", id="7"),  # will be filtered out
    ]

    # First summarization
    max_tokens = 6
    max_summary_tokens = 1
    result = await asummarize_messages(
        messages1,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
    )

    # Verify the first summarization result
    assert "summary" in result.messages[0].content.lower()
    assert len(result.messages) == 2
    assert result.messages[-1] == messages1[-1]
    assert len(model.invoke_calls) == 1

    # Check the summary value
    summary_value = result.running_summary
    assert summary_value.summary == "First summary of the conversation."
    assert len(summary_value.summarized_message_ids) == 6  # first 6 messages

    # Add more messages to trigger another summarization
    new_messages = [
        # these will be summarized (including accounting for the previous summary!)
        AIMessage(content="Response to latest 1", id="8"),  # will be filtered out
        HumanMessage(content="Message 4", id="9"),
        AIMessage(content="Response 4", id="10"),
        HumanMessage(content="Message 5", id="11"),
        AIMessage(content="Response 5", id="12"),
        # these will be kept in the final result
        HumanMessage(content="Message 6", id="13"),
        AIMessage(content="Response 6", id="14"),
        HumanMessage(content="Latest message 2", id="15"),
    ]

    # NOTE: here we're using the updated messages, not the original ones
    messages2 = result.messages.copy()
    messages2.extend(new_messages)

    # Second summarization
    result2 = await asummarize_messages(
        messages2,
        running_summary=summary_value,
        model=model,
        token_counter=len,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
    )

    # Check that model was called twice
    assert len(model.invoke_calls) == 2

    # Get the messages sent to the model in the second call
    second_call_messages = model.invoke_calls[1]

    # Check that the previous summary is included in the prompt
    prompt_message = second_call_messages[-1]
    assert "First summary of the conversation" in prompt_message.content
    assert "Extend this summary" in prompt_message.content

    # Check that only the new messages are sent to the model, not already summarized ones
    assert len(second_call_messages) == 5  # 4 messages + prompt
    assert [msg.content for msg in second_call_messages[:-1]] == [
        "Message 4",
        "Response 4",
        "Message 5",
        "Response 5",
    ]

    # Verify the structure of the final result
    assert "summary" in result2.messages[0].content.lower()
    assert len(result2.messages) == 4  # Summary + last 3 messages
    assert result2.messages[-3:] == messages2[-3:]

    # Check the updated summary
    updated_summary_value = result2.running_summary
    assert updated_summary_value.summary == "Updated summary including new messages."
    # Verify all messages except the last 3 were summarized
    assert len(updated_summary_value.summarized_message_ids) == 12


async def test_summarization_node():
    model = FakeChatModel(
        responses=[AIMessage(content="This is a summary of the conversation.")]
    )

    # Create enough messages to trigger summarization
    messages = [
        # these messages will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        # these messages will be added to the result post-summarization
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        HumanMessage(content="Latest message", id="9"),
    ]

    # Call the summarizer
    max_summary_tokens = 1
    summarization_node = SummarizationNode(
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=max_summary_tokens,
    )
    result = await summarization_node.ainvoke({"messages": messages})

    # Check that model was called
    assert len(model.invoke_calls) == 1

    # Check that the result has the expected structure:
    # - First message should be a summary
    # - Last 3 messages should be the last 3 original messages
    assert len(result["summarized_messages"]) == 4
    assert result["summarized_messages"][0].type == "system"
    assert "summary" in result["summarized_messages"][0].content.lower()
    assert result["summarized_messages"][1:] == messages[-3:]

    # Check the summary value
    summary_value = result["context"]["running_summary"]
    assert summary_value is not None
    assert summary_value.summary == "This is a summary of the conversation."
    assert summary_value.summarized_message_ids == set(
        msg.id for msg in messages[:6]
    )  # All messages except the latest

    # Test subsequent invocation (no new summary needed)
    result = await summarization_node.ainvoke({"messages": messages, "context": {"running_summary": summary_value}})
    assert len(result["summarized_messages"]) == 4
    assert result["summarized_messages"][0].type == "system"
    assert (
        result["summarized_messages"][0].content
        == "Summary of the conversation so far: This is a summary of the conversation."
    )
    assert result["summarized_messages"][1:] == messages[-3:]


async def test_summarization_node_same_key():
    # this is a variant of test_subsequent_summarization_with_new_messages
    # that passes the updated (ie., summarized) messages on the second turn
    model = FakeChatModel(
        responses=[
            AIMessage(content="First summary of the conversation."),
            AIMessage(content="Updated summary including new messages."),
        ]
    )

    # First batch of messages
    messages1 = [
        # these will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        # this will be propagated to the next summarization
        HumanMessage(content="Latest message 1", id="7"),  # will be filtered out
    ]

    # First summarization
    max_tokens = 6
    max_summary_tokens = 1
    summarization_node = SummarizationNode(
        model=model,
        token_counter=len,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
        input_messages_key="messages",
        output_messages_key="messages",
    )
    result = await summarization_node.ainvoke({"messages": messages1})

    # Verify the first summarization result
    assert result["messages"][0].type == "remove"
    assert "summary" in result["messages"][1].content.lower()
    assert len(result["messages"]) == 3
    assert result["messages"][-1] == messages1[-1]
    assert len(model.invoke_calls) == 1

    # Check the summary value
    summary_value = result["context"]["running_summary"]
    assert summary_value.summary == "First summary of the conversation."
    assert len(summary_value.summarized_message_ids) == 6  # first 6 messages

    # Add more messages to trigger another summarization
    new_messages = [
        # these will be summarized (including accounting for the previous summary!)
        AIMessage(content="Response to latest 1", id="8"),  # will be filtered out
        HumanMessage(content="Message 4", id="9"),
        AIMessage(content="Response 4", id="10"),
        HumanMessage(content="Message 5", id="11"),
        AIMessage(content="Response 5", id="12"),
        # these will be kept in the final result
        HumanMessage(content="Message 6", id="13"),
        AIMessage(content="Response 6", id="14"),
        HumanMessage(content="Latest message 2", id="15"),
    ]

    # NOTE: here we're using the updated messages, not the original ones
    messages2 = result["messages"][1:].copy()
    messages2.extend(new_messages)

    # Second summarization
    result2 = await summarization_node.ainvoke({"messages": messages2, "context": {"running_summary": summary_value}})

    # Check that model was called twice
    assert len(model.invoke_calls) == 2

    # Get the messages sent to the model in the second call
    second_call_messages = model.invoke_calls[1]

    # Check that the previous summary is included in the prompt
    prompt_message = second_call_messages[-1]
    assert "First summary of the conversation" in prompt_message.content
    assert "Extend this summary" in prompt_message.content

    # Check that only the new messages are sent to the model, not already summarized ones
    assert len(second_call_messages) == 5  # 4 messages + prompt
    assert [msg.content for msg in second_call_messages[:-1]] == [
        "Message 4",
        "Response 4",
        "Message 5",
        "Response 5"
    ]

    # Verify the structure of the final result
    assert result2["messages"][0].type == "remove"
    assert "summary" in result2["messages"][1].content.lower()
    assert len(result2["messages"]) == 5  # Remove message + summary + last 3 messages
    assert result2["messages"][-3:] == messages2[-3:]

    # Check the updated summary
    updated_summary_value = result2["context"]["running_summary"]
    assert updated_summary_value.summary == "Updated summary including new messages."
    # Verify all messages except the last 3 were summarized
    assert len(updated_summary_value.summarized_message_ids) == 12