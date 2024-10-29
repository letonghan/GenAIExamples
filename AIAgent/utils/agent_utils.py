
import json
from typing import Annotated, Sequence, TypedDict
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import BaseMessage
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep


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


def setup_model(llm_endpoint, model_id):
    print(f"---- set up model with: {llm_endpoint}")

    generation_params = {
        "max_new_tokens": 4096,
        "top_k": 10,
        "top_p": 0.95,
        "temperature": 0.001,
        "repetition_penalty": 1.03,
        "return_full_text": False,
        "streaming": True,
    }

    llm = HuggingFaceEndpoint(
        endpoint_url=llm_endpoint,
        task="text-generation",
        **generation_params,
    )

    my_model = ChatHuggingFace(llm=llm, model_id=model_id)
    return my_model

