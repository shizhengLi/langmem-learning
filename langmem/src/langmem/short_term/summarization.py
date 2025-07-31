import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, cast

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    MessageLikeRepresentation,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.prompts.chat import ChatPromptTemplate, ChatPromptValue
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]


DEFAULT_INITIAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{messages}"),
        ("user", "Create a summary of the conversation above:"),
    ]
)


DEFAULT_EXISTING_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{messages}"),
        (
            "user",
            "This is summary of the conversation so far: {existing_summary}\n\n"
            "Extend this summary by taking into account the new messages above:",
        ),
    ]
)

DEFAULT_FINAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        # if exists
        ("placeholder", "{system_message}"),
        ("system", "Summary of the conversation so far: {summary}"),
        ("placeholder", "{messages}"),
    ]
)


@dataclass
class RunningSummary:
    """Object for storing information about the previous summarization.

    Used on subsequent calls to summarize_messages to avoid summarizing the same messages.
    """

    summary: str
    """Latest summary of the messages, updated every time the summarization is performed."""

    summarized_message_ids: set[str]
    """The IDs of all of the messages that have been previously summarized."""

    last_summarized_message_id: str | None
    """The ID of the last message that was summarized."""


@dataclass
class SummarizationResult:
    """Result of message summarization."""

    messages: list[AnyMessage]
    """List of updated messages that are ready to be input to the LLM after summarization, including a message with a summary (if any)."""

    running_summary: RunningSummary | None = None
    """Information about previous summarization (the summary and the IDs of the previously summarized messages.
    Can be None if no summarization was performed (not enough messages to summarize).
    """


@dataclass
class PreprocessedMessages:
    """Container with messages to summarize and related bookkeeping information."""

    messages_to_summarize: list[AnyMessage]
    """Messages to summarize."""

    n_tokens_to_summarize: int
    """Number of tokens to summarize."""

    max_tokens_to_summarize: int
    """Maximum number of tokens to summarize."""

    total_summarized_messages: int
    """Total number of messages that have been summarized so far."""

    existing_system_message: SystemMessage | None
    """Existing system message (excluded from summarization)."""


def _preprocess_messages(
    *,
    messages: list[AnyMessage],
    running_summary: RunningSummary | None,
    max_tokens: int,
    max_tokens_before_summary: int | None,
    max_summary_tokens: int,
    token_counter: TokenCounter,
) -> PreprocessedMessages:
    """Preprocess messages for summarization."""
    if max_summary_tokens >= max_tokens:
        raise ValueError("`max_summary_tokens` must be less than `max_tokens`.")

    # Set max_tokens_before_summary to max_tokens if not provided
    if max_tokens_before_summary is None:
        max_tokens_before_summary = max_tokens

    max_tokens_to_summarize = max_tokens
    # Adjust the remaining token budget to account for the summary to be added
    max_remaining_tokens = max_tokens - max_summary_tokens
    # First handle system message if present
    if messages and isinstance(messages[0], SystemMessage):
        existing_system_message = messages[0]
        # remove the system message from the list of messages to summarize
        messages = messages[1:]
        # adjust the remaining token budget to account for the system message to be re-added
        max_remaining_tokens -= token_counter([existing_system_message])
    else:
        existing_system_message = None

    if not messages:
        return PreprocessedMessages(
            messages_to_summarize=[],
            n_tokens_to_summarize=0,
            max_tokens_to_summarize=max_tokens_to_summarize,
            total_summarized_messages=0,
            existing_system_message=existing_system_message,
        )

    # Get previously summarized messages, if any
    summarized_message_ids = set()
    total_summarized_messages = 0
    if running_summary:
        summarized_message_ids = running_summary.summarized_message_ids
        # Adjust the summarization token budget to account for the previous summary
        max_tokens_to_summarize -= token_counter(
            [SystemMessage(content=running_summary.summary)]
        )
        # If we have an existing running summary, find how many messages have been
        # summarized so far based on the last summarized message ID.
        for i, message in enumerate(messages):
            if message.id == running_summary.last_summarized_message_id:
                total_summarized_messages = i + 1
                break

    # We will use this to ensure that the total number of resulting tokens
    # will fit into max_tokens window.
    total_n_tokens = token_counter(messages[total_summarized_messages:])

    # Go through messages to count tokens and find cutoff point
    n_tokens = 0
    idx = max(0, total_summarized_messages - 1)
    # map tool call IDs to their corresponding tool messages
    tool_call_id_to_tool_message: dict[str, ToolMessage] = {}
    should_summarize = False
    n_tokens_to_summarize = 0
    for i in range(total_summarized_messages, len(messages)):
        message = messages[i]
        if message.id is None:
            raise ValueError("Messages are required to have ID field.")

        if message.id in summarized_message_ids:
            raise ValueError(
                f"Message with ID {message.id} has already been summarized."
            )

        # Store tool messages by their tool_call_id for later reference
        if isinstance(message, ToolMessage) and message.tool_call_id:
            tool_call_id_to_tool_message[message.tool_call_id] = message

        n_tokens += token_counter([message])

        # Check if we've reached max_tokens_to_summarize
        # and the remaining messages fit within the max_remaining_tokens budget
        if (
            n_tokens >= max_tokens_before_summary
            and total_n_tokens - n_tokens <= max_remaining_tokens
            and not should_summarize
        ):
            n_tokens_to_summarize = n_tokens
            should_summarize = True
            idx = i

    # Note: we don't return here since we might still need to include the existing summary
    if not should_summarize:
        messages_to_summarize = []
    else:
        messages_to_summarize = messages[total_summarized_messages : idx + 1]

    # If the last message is an AI message with tool calls,
    # include subsequent corresponding tool messages in the summary as well,
    # to avoid issues w/ the LLM provider
    if (
        messages_to_summarize
        and isinstance(messages_to_summarize[-1], AIMessage)
        and (tool_calls := messages_to_summarize[-1].tool_calls)
    ):
        # Add any matching tool messages from our dictionary
        for tool_call in tool_calls:
            if tool_call["id"] in tool_call_id_to_tool_message:
                tool_message = tool_call_id_to_tool_message[tool_call["id"]]
                n_tokens_to_summarize += token_counter([tool_message])
                messages_to_summarize.append(tool_message)

    return PreprocessedMessages(
        messages_to_summarize=messages_to_summarize,
        n_tokens_to_summarize=n_tokens_to_summarize,
        max_tokens_to_summarize=max_tokens_to_summarize,
        total_summarized_messages=total_summarized_messages,
        existing_system_message=existing_system_message,
    )


def _adjust_messages_before_summarization(
    preprocessed_messages: PreprocessedMessages, token_counter: TokenCounter
) -> list[AnyMessage]:
    # Check if the number of tokens to summarize exceeds max_tokens.
    # If it does, filter out the oldest messages to make it fit within max_tokens.
    # The reason we do this is to ensure that we don't exceed context window of the summarization LLM,
    # and we make an assumption that the same model is used both for the underlying app and for summarization.
    if (
        preprocessed_messages.n_tokens_to_summarize
        > preprocessed_messages.max_tokens_to_summarize
    ):
        adjusted_messages_to_summarize = trim_messages(
            preprocessed_messages.messages_to_summarize,
            # TODO: consider exposing max_tokens_to_summarize as a separate parameter
            max_tokens=preprocessed_messages.max_tokens_to_summarize,
            token_counter=token_counter,
            start_on="human",
            strategy="last",
            allow_partial=True,
        )
        if not adjusted_messages_to_summarize:
            warnings.warn(
                "Failed to trim messages to fit within max_tokens limit before summarization - "
                "falling back to the original message list. "
                "This may lead to exceeding the context window of the summarization LLM.",
                RuntimeWarning,
            )
            adjusted_messages_to_summarize = preprocessed_messages.messages_to_summarize
    else:
        adjusted_messages_to_summarize = preprocessed_messages.messages_to_summarize

    return adjusted_messages_to_summarize


def _prepare_input_to_summarization_model(
    *,
    preprocessed_messages: PreprocessedMessages,
    running_summary: RunningSummary | None,
    existing_summary_prompt: ChatPromptTemplate,
    initial_summary_prompt: ChatPromptTemplate,
    token_counter: TokenCounter,
) -> list[AnyMessage]:
    adjusted_messages_to_summarize = _adjust_messages_before_summarization(
        preprocessed_messages, token_counter
    )
    if running_summary:
        summary_messages = cast(
            ChatPromptValue,
            existing_summary_prompt.invoke(
                {
                    "messages": adjusted_messages_to_summarize,
                    "existing_summary": running_summary.summary,
                }
            ),
        )
    else:
        summary_messages = cast(
            ChatPromptValue,
            initial_summary_prompt.invoke({"messages": adjusted_messages_to_summarize}),
        )

    return summary_messages.messages


def _prepare_summarization_result(
    *,
    preprocessed_messages: PreprocessedMessages,
    messages: list[AnyMessage],
    existing_summary: RunningSummary | None,
    running_summary: RunningSummary | None,
    final_prompt: ChatPromptTemplate,
) -> SummarizationResult:
    total_summarized_messages = preprocessed_messages.total_summarized_messages + len(
        preprocessed_messages.messages_to_summarize
    )
    if running_summary:
        # Only include system message if it doesn't overlap with the existing summary.
        # This is useful if the messages passed to summarize_messages already include a system message with summary.
        # This usually happens when summarization node overwrites the message history.
        include_system_message = preprocessed_messages.existing_system_message and not (
            existing_summary
            and existing_summary.summary
            in preprocessed_messages.existing_system_message.content
        )
        updated_messages = cast(
            ChatPromptValue,
            final_prompt.invoke(
                {
                    "system_message": [preprocessed_messages.existing_system_message]
                    if include_system_message
                    else [],
                    "summary": running_summary.summary,
                    "messages": messages[total_summarized_messages:],
                }
            ),
        )
        return SummarizationResult(
            running_summary=running_summary,
            messages=updated_messages.messages,
        )
    else:
        # no changes are needed
        return SummarizationResult(
            running_summary=None,
            messages=(
                messages
                if preprocessed_messages.existing_system_message is None
                else [preprocessed_messages.existing_system_message] + messages
            ),
        )


def summarize_messages(
    messages: list[AnyMessage],
    *,
    running_summary: RunningSummary | None,
    model: LanguageModelLike,
    max_tokens: int,
    max_tokens_before_summary: int | None = None,
    max_summary_tokens: int = 256,
    token_counter: TokenCounter = count_tokens_approximately,
    initial_summary_prompt: ChatPromptTemplate = DEFAULT_INITIAL_SUMMARY_PROMPT,
    existing_summary_prompt: ChatPromptTemplate = DEFAULT_EXISTING_SUMMARY_PROMPT,
    final_prompt: ChatPromptTemplate = DEFAULT_FINAL_SUMMARY_PROMPT,
) -> SummarizationResult:
    """Summarize messages when they exceed a token limit and replace them with a summary message.

    This function processes the messages from oldest to newest: once the cumulative number of message tokens
    reaches `max_tokens_before_summary`, all messages within `max_tokens_before_summary` are summarized (excluding the system message, if any)
    and replaced with a new summary message. The resulting list of messages is [summary_message] + remaining_messages.

    Args:
        messages: The list of messages to process.
        running_summary: Optional running summary object with information about the previous summarization. If provided:
            - only messages that were **not** previously summarized will be processed
            - if no new summary is generated, the running summary will be added to the returned messages
            - if a new summary needs to be generated, it is generated by incorporating the existing summary value from the running summary
        model: The language model to use for generating summaries.
        max_tokens: Maximum number of tokens to return in the final output. Will be enforced only after summarization.
            This will also be used as the maximum number of tokens to feed to the summarization LLM.
        max_tokens_before_summary: Maximum number of tokens to accumulate before triggering summarization.
            Defaults to the same value as `max_tokens` if not provided.
            This allows fitting more tokens into the summarization LLM, if needed.

            !!! Note

                If the last message within `max_tokens_before_summary` is an AI message with tool calls,
                all of the subsequent, corresponding tool messages will be summarized as well.

            !!! Note

                If the number of tokens to be summarized is greater than max_tokens, only the last max_tokens amongst those
                will be summarized. This is done to prevent exceeding the context window of the summarization LLM
                (assumed to be capped at max_tokens).
        max_summary_tokens: Maximum number of tokens to budget for the summary.

            !!! Note

                This parameter is not passed to the summary-generating LLM to limit the length of the summary.
                It is only used for correctly estimating the maximum allowed token budget.
                If you want to enforce it, you would need to pass `model.bind(max_tokens=max_summary_tokens)`
                as the `model` parameter to this function.
        token_counter: Function to count tokens in a message. Defaults to approximate counting.
            For more accurate counts you can use `model.get_num_tokens_from_messages`.
        initial_summary_prompt: Prompt template for generating the first summary.
        existing_summary_prompt: Prompt template for updating an existing (running) summary.
        final_prompt: Prompt template that combines summary with the remaining messages before returning.

    Returns:
        A SummarizationResult object containing the updated messages and a running summary.
            - messages: list of updated messages ready to be input to the LLM
            - running_summary: RunningSummary object
                - summary: text of the latest summary
                - summarized_message_ids: set of message IDs that were previously summarized
                - last_summarized_message_id: ID of the last message that was summarized

    Example:
        ```pycon
        from langgraph.graph import StateGraph, START, MessagesState
        from langgraph.checkpoint.memory import InMemorySaver
        from langmem.short_term import summarize_messages, RunningSummary
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o")
        summarization_model = model.bind(max_tokens=128)


        class SummaryState(MessagesState):
            summary: RunningSummary | None


        def call_model(state):
            summarization_result = summarize_messages(
                state["messages"],
                running_summary=state.get("summary"),
                model=summarization_model,
                max_tokens=256,
                max_tokens_before_summary=256,
                max_summary_tokens=128,
            )
            response = model.invoke(summarization_result.messages)
            state_update = {"messages": [response]}
            if summarization_result.running_summary:
                state_update["summary"] = summarization_result.running_summary
            return state_update


        checkpointer = InMemorySaver()
        workflow = StateGraph(SummaryState)
        workflow.add_node(call_model)
        workflow.add_edge(START, "call_model")
        graph = workflow.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "1"}}
        graph.invoke({"messages": "hi, my name is bob"}, config)
        graph.invoke({"messages": "write a short poem about cats"}, config)
        graph.invoke({"messages": "now do the same but for dogs"}, config)
        graph.invoke({"messages": "what's my name?"}, config)
        ```
    """
    preprocessed_messages = _preprocess_messages(
        messages=messages,
        running_summary=running_summary,
        max_tokens=max_tokens,
        max_tokens_before_summary=max_tokens_before_summary,
        max_summary_tokens=max_summary_tokens,
        token_counter=token_counter,
    )
    if preprocessed_messages.existing_system_message:
        messages = messages[1:]

    if not messages:
        return SummarizationResult(
            running_summary=running_summary,
            messages=(
                messages
                if preprocessed_messages.existing_system_message is None
                else [preprocessed_messages.existing_system_message] + messages
            ),
        )

    existing_summary = running_summary
    summarized_message_ids = (
        set(running_summary.summarized_message_ids) if running_summary else set()
    )
    if preprocessed_messages.messages_to_summarize:
        summary_messages = _prepare_input_to_summarization_model(
            preprocessed_messages=preprocessed_messages,
            running_summary=running_summary,
            existing_summary_prompt=existing_summary_prompt,
            initial_summary_prompt=initial_summary_prompt,
            token_counter=token_counter,
        )
        summary_response = model.invoke(summary_messages)
        summarized_message_ids = summarized_message_ids | set(
            message.id for message in preprocessed_messages.messages_to_summarize
        )
        running_summary = RunningSummary(
            summary=summary_response.content,
            summarized_message_ids=summarized_message_ids,
            last_summarized_message_id=preprocessed_messages.messages_to_summarize[
                -1
            ].id,
        )

    return _prepare_summarization_result(
        preprocessed_messages=preprocessed_messages,
        messages=messages,
        existing_summary=existing_summary,
        running_summary=running_summary,
        final_prompt=final_prompt,
    )


async def asummarize_messages(
    messages: list[AnyMessage],
    *,
    running_summary: RunningSummary | None,
    model: LanguageModelLike,
    max_tokens: int,
    max_tokens_before_summary: int | None = None,
    max_summary_tokens: int = 256,
    token_counter: TokenCounter = count_tokens_approximately,
    initial_summary_prompt: ChatPromptTemplate = DEFAULT_INITIAL_SUMMARY_PROMPT,
    existing_summary_prompt: ChatPromptTemplate = DEFAULT_EXISTING_SUMMARY_PROMPT,
    final_prompt: ChatPromptTemplate = DEFAULT_FINAL_SUMMARY_PROMPT,
) -> SummarizationResult:
    """Summarize messages asynchronously when they exceed a token limit and replace them with a summary message.

    This function processes the messages from oldest to newest: once the cumulative number of message tokens
    reaches `max_tokens_before_summary`, all messages within `max_tokens_before_summary` are summarized (excluding the system message, if any)
    and replaced with a new summary message. The resulting list of messages is [summary_message] + remaining_messages.

    Args:
        messages: The list of messages to process.
        running_summary: Optional running summary object with information about the previous summarization. If provided:
            - only messages that were **not** previously summarized will be processed
            - if no new summary is generated, the running summary will be added to the returned messages
            - if a new summary needs to be generated, it is generated by incorporating the existing summary value from the running summary
        model: The language model to use for generating summaries.
        max_tokens: Maximum number of tokens to return in the final output. Will be enforced only after summarization.
        max_tokens_before_summary: Maximum number of tokens to accumulate before triggering summarization.
            Defaults to the same value as `max_tokens` if not provided.
            This allows fitting more tokens into the summarization LLM, if needed.

            !!! Note

                If the last message within `max_tokens_before_summary` is an AI message with tool calls,
                all of the subsequent, corresponding tool messages will be summarized as well.

            !!! Note

                If the number of tokens to be summarized is greater than max_tokens, only the last max_tokens amongst those
                will be summarized. This is done to prevent exceeding the context window of the summarization LLM
                (assumed to be capped at max_tokens).
        max_summary_tokens: Maximum number of tokens to budget for the summary.

            !!! Note

                This parameter is not passed to the summary-generating LLM to limit the length of the summary.
                It is only used for correctly estimating the maximum allowed token budget.
                If you want to enforce it, you would need to pass `model.bind(max_tokens=max_summary_tokens)`
                as the `model` parameter to this function.
        token_counter: Function to count tokens in a message. Defaults to approximate counting.
            For more accurate counts you can use `model.get_num_tokens_from_messages`.
        initial_summary_prompt: Prompt template for generating the first summary.
        existing_summary_prompt: Prompt template for updating an existing (running) summary.
        final_prompt: Prompt template that combines summary with the remaining messages before returning.

    Returns:
        A SummarizationResult object containing the updated messages and a running summary.
            - messages: list of updated messages ready to be input to the LLM
            - running_summary: RunningSummary object
                - summary: text of the latest summary
                - summarized_message_ids: set of message IDs that were previously summarized
                - last_summarized_message_id: ID of the last message that was summarized

    Example:
        ```pycon
        from langgraph.graph import StateGraph, START, MessagesState
        from langgraph.checkpoint.memory import InMemorySaver
        from langmem.short_term import asummarize_messages, RunningSummary
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o")
        summarization_model = model.bind(max_tokens=128)


        class SummaryState(MessagesState):
            summary: RunningSummary | None


        async def call_model(state):
            summarization_result = await asummarize_messages(
                state["messages"],
                running_summary=state.get("summary"),
                model=summarization_model,
                max_tokens=256,
                max_tokens_before_summary=256,
                max_summary_tokens=128,
            )
            response = await model.ainvoke(summarization_result.messages)
            state_update = {"messages": [response]}
            if summarization_result.running_summary:
                state_update["summary"] = summarization_result.running_summary
            return state_update


        checkpointer = InMemorySaver()
        workflow = StateGraph(SummaryState)
        workflow.add_node(call_model)
        workflow.add_edge(START, "call_model")
        graph = workflow.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "1"}}
        await graph.ainvoke({"messages": "hi, my name is bob"}, config)
        await graph.ainvoke({"messages": "write a short poem about cats"}, config)
        await graph.ainvoke({"messages": "now do the same but for dogs"}, config)
        await graph.ainvoke({"messages": "what's my name?"}, config)
        ```
    """
    preprocessed_messages = _preprocess_messages(
        messages=messages,
        running_summary=running_summary,
        max_tokens=max_tokens,
        max_tokens_before_summary=max_tokens_before_summary,
        max_summary_tokens=max_summary_tokens,
        token_counter=token_counter,
    )
    if preprocessed_messages.existing_system_message:
        messages = messages[1:]

    if not messages:
        return SummarizationResult(
            running_summary=running_summary,
            messages=(
                messages
                if preprocessed_messages.existing_system_message is None
                else [preprocessed_messages.existing_system_message] + messages
            ),
        )

    existing_summary = running_summary
    summarized_message_ids = (
        set(running_summary.summarized_message_ids) if running_summary else set()
    )
    if preprocessed_messages.messages_to_summarize:
        summary_messages = _prepare_input_to_summarization_model(
            preprocessed_messages=preprocessed_messages,
            running_summary=running_summary,
            existing_summary_prompt=existing_summary_prompt,
            initial_summary_prompt=initial_summary_prompt,
            token_counter=token_counter,
        )
        summary_response = await model.ainvoke(summary_messages)
        summarized_message_ids = summarized_message_ids | set(
            message.id for message in preprocessed_messages.messages_to_summarize
        )
        running_summary = RunningSummary(
            summary=summary_response.content,
            summarized_message_ids=summarized_message_ids,
            last_summarized_message_id=preprocessed_messages.messages_to_summarize[
                -1
            ].id,
        )

    return _prepare_summarization_result(
        preprocessed_messages=preprocessed_messages,
        messages=messages,
        existing_summary=existing_summary,
        running_summary=running_summary,
        final_prompt=final_prompt,
    )


class SummarizationNode(RunnableCallable):
    """A LangGraph node that summarizes messages when they exceed a token limit and replaces them with a summary message."""

    def __init__(
        self,
        *,
        model: LanguageModelLike,
        max_tokens: int,
        max_tokens_before_summary: int | None = None,
        max_summary_tokens: int = 256,
        token_counter: TokenCounter = count_tokens_approximately,
        initial_summary_prompt: ChatPromptTemplate = DEFAULT_INITIAL_SUMMARY_PROMPT,
        existing_summary_prompt: ChatPromptTemplate = DEFAULT_EXISTING_SUMMARY_PROMPT,
        final_prompt: ChatPromptTemplate = DEFAULT_FINAL_SUMMARY_PROMPT,
        input_messages_key: str = "messages",
        output_messages_key: str = "summarized_messages",
        name: str = "summarization",
    ) -> None:
        """A LangGraph node that summarizes messages when they exceed a token limit and replaces them with a summary message.

        Processes the messages from oldest to newest: once the cumulative number of message tokens
        reaches `max_tokens_before_summary`, all messages within `max_tokens_before_summary` are summarized (excluding the system message, if any)
        and replaced with a new summary message. The resulting list of messages is [summary_message] + remaining_messages.

        Args:
            model: The language model to use for generating summaries.
            max_tokens: Maximum number of tokens to return in the final output. Will be enforced only after summarization.
            max_tokens_before_summary: Maximum number of tokens to accumulate before triggering summarization.
                Defaults to the same value as `max_tokens` if not provided.
                This allows fitting more tokens into the summarization LLM, if needed.

                !!! Note

                    If the last message within `max_tokens_before_summary` is an AI message with tool calls,
                    all of the subsequent, corresponding tool messages will be summarized as well.

                !!! Note

                    If the number of tokens to be summarized is greater than max_tokens, only the last max_tokens amongst those
                    will be summarized. This is done to prevent exceeding the context window of the summarization LLM
                    (assumed to be capped at max_tokens).
            max_summary_tokens: Maximum number of tokens to budget for the summary.

                !!! Note

                    This parameter is not passed to the summary-generating LLM to limit the length of the summary.
                    It is only used for correctly estimating the maximum allowed token budget.
                    If you want to enforce it, you would need to pass `model.bind(max_tokens=max_summary_tokens)`
                    as the `model` parameter to this function.
            token_counter: Function to count tokens in a message. Defaults to approximate counting.
                For more accurate counts you can use `model.get_num_tokens_from_messages`.
            initial_summary_prompt: Prompt template for generating the first summary.
            existing_summary_prompt: Prompt template for updating an existing (running) summary.
            final_prompt: Prompt template that combines summary with the remaining messages before returning.
            input_messages_key: Key in the input graph state that contains the list of messages to summarize.
            output_messages_key: Key in the state update that contains the list of updated messages.
                !!! Warning

                    By default, the `output_messages_key` **is different** from the `input_messages_key`.
                    This is done to decouple summarized messages from the main list of messages in the graph state (i.e., `input_messages_key`).
                    You should only make them the same if you want to **overwrite** the main list of messages (i.e., `input_messages_key`).

            name: Name of the summarization node.

        Returns:
            LangGraph state update in the following format:
                ```json
                {
                    "output_messages_key": <list of updated messages ready to be input to the LLM after summarization, including a message with a summary (if any)>,
                    "context": {"running_summary": <RunningSummary object>}
                }
                ```

        Example:
            ```pycon
            from typing import Any, TypedDict
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import AnyMessage
            from langgraph.graph import StateGraph, START, MessagesState
            from langgraph.checkpoint.memory import InMemorySaver
            from langmem.short_term import SummarizationNode, RunningSummary

            model = ChatOpenAI(model="gpt-4o")
            summarization_model = model.bind(max_tokens=128)


            class State(MessagesState):
                context: dict[str, Any]


            class LLMInputState(TypedDict):
                summarized_messages: list[AnyMessage]
                context: dict[str, Any]


            summarization_node = SummarizationNode(
                model=summarization_model,
                max_tokens=256,
                max_tokens_before_summary=256,
                max_summary_tokens=128,
            )


            def call_model(state: LLMInputState):
                response = model.invoke(state["summarized_messages"])
                return {"messages": [response]}


            checkpointer = InMemorySaver()
            workflow = StateGraph(State)
            workflow.add_node(call_model)
            workflow.add_node("summarize", summarization_node)
            workflow.add_edge(START, "summarize")
            workflow.add_edge("summarize", "call_model")
            graph = workflow.compile(checkpointer=checkpointer)

            config = {"configurable": {"thread_id": "1"}}
            graph.invoke({"messages": "hi, my name is bob"}, config)
            graph.invoke({"messages": "write a short poem about cats"}, config)
            graph.invoke({"messages": "now do the same but for dogs"}, config)
            graph.invoke({"messages": "what's my name?"}, config)
            ```
        """
        super().__init__(self._func, self._afunc, name=name, trace=False)
        self.model = model
        self.max_tokens = max_tokens
        self.max_tokens_before_summary = max_tokens_before_summary
        self.max_summary_tokens = max_summary_tokens
        self.token_counter = token_counter
        self.initial_summary_prompt = initial_summary_prompt
        self.existing_summary_prompt = existing_summary_prompt
        self.final_prompt = final_prompt
        self.input_messages_key = input_messages_key
        self.output_messages_key = output_messages_key

    def _parse_input(
        self, input: dict[str, Any] | BaseModel
    ) -> tuple[list[AnyMessage], dict[str, Any]]:
        if isinstance(input, dict):
            messages = input.get(self.input_messages_key)
            context = input.get("context", {})
        elif isinstance(input, BaseModel):
            messages = getattr(input, self.input_messages_key, None)
            context = getattr(input, "context", {})
        else:
            raise ValueError(f"Invalid input type: {type(input)}")

        if messages is None:
            raise ValueError(
                f"Missing required field `{self.input_messages_key}` in the input."
            )
        return messages, context

    def _prepare_state_update(
        self, context: dict[str, Any], summarization_result: SummarizationResult
    ) -> dict[str, Any]:
        state_update = {self.output_messages_key: summarization_result.messages}
        if summarization_result.running_summary:
            state_update["context"] = {
                **context,
                "running_summary": summarization_result.running_summary,
            }
            # If the input and output messages keys are the same, we need to remove the
            # summarized messages from the resulting message list
            if self.input_messages_key == self.output_messages_key:
                state_update[self.output_messages_key] = [
                    RemoveMessage(REMOVE_ALL_MESSAGES)
                ] + state_update[self.output_messages_key]
        return state_update

    def _func(self, input: dict[str, Any] | BaseModel) -> dict[str, Any]:
        messages, context = self._parse_input(input)
        summarization_result = summarize_messages(
            messages,
            running_summary=context.get("running_summary"),
            model=self.model,
            max_tokens=self.max_tokens,
            max_tokens_before_summary=self.max_tokens_before_summary,
            max_summary_tokens=self.max_summary_tokens,
            token_counter=self.token_counter,
            initial_summary_prompt=self.initial_summary_prompt,
            existing_summary_prompt=self.existing_summary_prompt,
            final_prompt=self.final_prompt,
        )
        return self._prepare_state_update(context, summarization_result)

    async def _afunc(self, input: dict[str, Any] | BaseModel) -> dict[str, Any]:
        messages, context = self._parse_input(input)
        summarization_result = await asummarize_messages(
            messages,
            running_summary=context.get("running_summary"),
            model=self.model,
            max_tokens=self.max_tokens,
            max_tokens_before_summary=self.max_tokens_before_summary,
            max_summary_tokens=self.max_summary_tokens,
            token_counter=self.token_counter,
            initial_summary_prompt=self.initial_summary_prompt,
            existing_summary_prompt=self.existing_summary_prompt,
            final_prompt=self.final_prompt,
        )
        return self._prepare_state_update(context, summarization_result)
