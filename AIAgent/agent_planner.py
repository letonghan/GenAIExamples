
import json
from langchain import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from prompts import REACT_AGENT_LLAMA_PROMPT
from utils.agent_utils import (
    setup_chat_model,
    AgentState,
    AgentOutputParser,
)
from utils.utils import (
    get_tools_descriptions,
    tool_renderer,
    assemble_history,
    convert_json_to_tool_call,
    get_args,
    extract_web_source
)


class AgentNode:
    
    def __init__(self, llm_endpoint, model_id, tool_yaml_path, language):
        # init llm chain
        self.prompt = PromptTemplate(
            template=REACT_AGENT_LLAMA_PROMPT,
            input_variables=["input", "history", "tools", "language"],
        )
        self.my_model = setup_chat_model(llm_endpoint, model_id)
        output_parser = AgentOutputParser()
        
        self.chain = self.prompt | self.my_model | output_parser
        self.tool_yaml_path = tool_yaml_path
        self.language = language
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
        output = self.chain.invoke({
            "input": query, 
            "history": history, 
            "tools": self.tools_descriptions,
            "language": self.language
        })

        # convert output to tool calls
        tool_calls = []
        for res in output:
            if "tool" in res:
                add_kw_tc, tool_call = convert_json_to_tool_call(res)
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
            tool_yaml_path=args.tool_yaml_path,
            language=args.language
        )
        tool_node = ToolNode(self.tools_descriptions)
        workflow = StateGraph(AgentState)
        
        # Define the nodes we will cycle between
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")
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
                    for k, v in node_state.items():
                        if v is not None:
                            if node_name == "agent":
                                if v[0].content == "":
                                    tool_name = v[0].additional_kwargs['tool_calls'][0].function.name
                                    result = {"tool": tool_name}
                                else:
                                    print(v[0].content)
                                    result = {"content": v[0].content.replace("\n", " ")}
                                yield f"data: {json.dumps(result)}\n\n"
                            elif node_name == "tools":
                                full_content = v[0].content
                                tool_name = v[0].name
                                result = {
                                    "tool": tool_name,
                                    "content": full_content
                                }
                                if "web" in tool_name:
                                    source = extract_web_source(full_content)
                                    result["source"] = source
                                yield f"data: {json.dumps(result)}\n\n"
                                
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield str(e)
            
    async def non_streaming_run(self, query, config):
        initial_state = self.prepare_initial_state(query)
        try:
            async for s in self.app.astream(initial_state, config=config, stream_mode="values"):
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

