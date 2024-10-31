
import json
from typing import Annotated, Sequence, TypedDict
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langchain_openai import ChatOpenAI


class AgentOutputParser(BaseOutputParser):
    def parse(self, text: str):
        json_lines = text.split("\n")
        output = []
        for line in json_lines:
            try:
                if "assistant" in line:
                    line = line.replace("assistant", "")
                output.append(json.loads(line))
            except Exception as e:
                # parsed text is not json, return text directly
                print("Exception happened in output parsing: ", str(e))
        if output:
            return output
        else:
            return text


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep


def setup_chat_model(llm_endpoint, model_id):
    openai_endpoint = f"{llm_endpoint}/v1"
    params = {
        "temperature": 0.1,
        "max_tokens": 1024,
        "streaming": False,
    }
    llm = ChatOpenAI(openai_api_key="EMPTY", openai_api_base=openai_endpoint, model_name=model_id, **params)
    return llm