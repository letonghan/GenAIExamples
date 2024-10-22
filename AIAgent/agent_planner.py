# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from langchain import PromptTemplate
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from huggingface_hub import ChatCompletionOutputFunctionDefinition, ChatCompletionOutputToolCall
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from prompts import REACT_AGENT_LLAMA_PROMPT
from utils.agent_utils import (
    setup_model,
    AgentState,
    AgentOutputParser,
)
from utils.utils import (
    get_tools_descriptions,
    tool_renderer,
    assemble_history,
    convert_json_to_tool_call,
    get_args
)


class AgentNode:
    
    def __init__(self, llm_endpoint, model_id, tool_yaml_path):
        # init llm chain
        prompt = PromptTemplate(
            template=REACT_AGENT_LLAMA_PROMPT,
            input_variables=["input", "history", "tools"],
        )
        my_model = setup_model(llm_endpoint, model_id)
        output_parser = AgentOutputParser()
        
        self.chain = prompt | my_model | output_parser
        self.tool_yaml_path = tool_yaml_path
        self._load_tools()
        
    def _load_tools(self):
        if not self.tool_yaml_path:
            pass
        self.tools = get_tools_descriptions(self.tool_yaml_path)
        self.tools_descriptions = tool_renderer(self.tools)
        
    def __call__(self, state):
        # prepare input parameters
        messages = state["messages"]
        query = messages[0].content
        history = assemble_history(messages)
        
        # invoke llm chain
        output = self.chain.invoke({"input": query, "history": history, "tools": self.tools_descriptions})
            
        # convert output to tool calls
        tool_calls = []
        for res in output:
            if "tool" in res:
                add_kw_tc, tool_call = convert_json_to_tool_call(res)
                # print("Tool call:\n", tool_call)
                tool_calls.append(tool_call)

        if tool_calls:
            ai_message = AIMessage(content="", additional_kwargs=add_kw_tc, tool_calls=tool_calls)
        elif "answer" in output[0]:
            ai_message = AIMessage(content=output[0]["answer"])
        else:
            ai_message = AIMessage(content=output)
        return {"messages": [ai_message]}


class AgentPlanner:
    
    def __init__(self, args) -> None:
        self.tools_descriptions = get_tools_descriptions(args.tool_yaml_path)
        agent_node = AgentNode(
            llm_endpoint=args.llm_endpoint_url,
            model_id=args.model,
            tool_yaml_path=args.tool_yaml_path
        )
        tool_node = ToolNode(self.tools_descriptions)
        workflow = StateGraph(AgentState)
        
        # Define the nodes we will cycle between
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
            {
                # If `tools`, then we call the tool node.
                "continue": "tools",
                # Otherwise we finish.
                "end": END,
            },
        )

        # We now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        workflow.add_edge("tools", "agent")

        self.app = workflow.compile()
        
    def should_continue(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"
        
    def prepare_initial_state(self, query):
        return {"messages": [HumanMessage(content=query)]}
    
    async def stream_generator(self, query, config):
        initial_state = self.prepare_initial_state(query)
        try:
            async for event in self.app.astream(initial_state, config=config):
                for node_name, node_state in event.items():
                    yield f"--- CALL {node_name} ---\n"
                    for k, v in node_state.items():
                        if v is not None:
                            yield f"{k}: {v}\n"

                yield f"data: {repr(event)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield str(e)
            
    async def non_streaming_run(self, query, config):
        initial_state = self.prepare_initial_state(query)
        print(f"################# planner non streaming run")
        try:
            async for s in self.app.astream(initial_state, config=config, stream_mode="values"):
                print(s)
                print("-------------------")
                message = s["messages"][-1]
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()

            last_message = s["messages"][-1]
            print("******Response: ", last_message.content)
            return last_message.content
        except Exception as e:
            return str(e)


async def main():
    args, _ = get_args()
    planner = AgentPlanner(args)
    input_query = "Most recent album by Taylor Swift"
    config = {"recursion_limit": args.recursion_limit}

    response = await planner.non_streaming_run(input_query, config)
    print("============ finish ===========")
    print(response)


asyncio.run(main())


response = {
    'messages': 
        [
            HumanMessage(
                content='Most recent album by Taylor Swift', 
                additional_kwargs={}, 
                response_metadata={}, 
                id='56b41a69-a5d5-4b86-882c-39a5675ff85c'), 
            AIMessage(
                content='', 
                additional_kwargs={
                    'tool_calls': [
                        ChatCompletionOutputToolCall(
                            function=ChatCompletionOutputFunctionDefinition(
                                arguments={'query': 'Taylor Swift most recent album'}, 
                                name='search_knowledge_base', 
                                description=None), 
                            id='9ca7f38f-91ac-4c74-a9b1-ae21023bd4c9', 
                            type='function')]}, 
                response_metadata={}, 
                id='059ee25a-0a55-4316-91fa-288edef720db', 
                tool_calls=[{
                    'name': 'search_knowledge_base', 
                    'args': {'query': 'Taylor Swift most recent album'}, 
                    'id': '9ca7f38f-91ac-4c74-a9b1-ae21023bd4c9', 
                    'type': 'tool_call'}]), 
            ToolMessage(
                content="Taylor Swift's most recent album is 'Midnights', which was released on October 21, 2022.", 
                name='search_knowledge_base', 
                id='669800c6-21c8-43b6-8650-16f289835343', 
                tool_call_id='9ca7f38f-91ac-4c74-a9b1-ae21023bd4c9'), 
            AIMessage(
                content="Taylor Swift's most recent album is 'Midnights', which was released on October 21, 2022.", 
                additional_kwargs={}, 
                response_metadata={}, 
                id='8659b3e6-4dc4-4dab-9aac-465b38e18e96')
        ]
}

