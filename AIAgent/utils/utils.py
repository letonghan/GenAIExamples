
import os
import re
import glob
import yaml
import uuid
import argparse
import importlib
from pydantic import BaseModel, Field, create_model
from langchain.tools import BaseTool, StructuredTool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from huggingface_hub import ChatCompletionOutputFunctionDefinition, ChatCompletionOutputToolCall
from config import env_config


def extract_task_list(llm_output: str) -> list:
    """
    This function extracts answers from the LLM output which follows the format:
    query: "<query_text>", answer: ["<answer_text>"]
    
    Args:
        llm_output (str): The text generated by the LLM which contains queries and answers.
    
    Returns:
        list: A list of extracted answers from the LLM output.
    """
    pattern = r'answer: \[(.*?)\]'
    
    matches = re.findall(pattern, llm_output)
    
    task_list = []
    for match in matches:
        task_list.extend(match.replace("\"", "").split(", "))
    
    return [s.replace(" ", "") for s in task_list]


def get_args():
    parser = argparse.ArgumentParser()
    # llm args
    parser.add_argument("--streaming", type=str, default="true")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--recursion_limit", type=int, default=5)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--llm_endpoint_url", type=str, default="http://localhost:8080")
    parser.add_argument("--tool_yaml_path", type=str, default="./tools/supervisor_agent_tools.yaml")
    parser.add_argument("--language", type=str, default="Chinese")

    sys_args, unknown_args = parser.parse_known_args()
    if env_config != []:
        env_args, env_unknown_args = parser.parse_known_args(env_config)
        unknown_args += env_unknown_args
        for key, value in vars(env_args).items():
            setattr(sys_args, key, value)

    if sys_args.streaming == "true":
        sys_args.streaming = True
    else:
        sys_args.streaming = False
    print("==========sys_args==========:\n", sys_args)
    return sys_args, unknown_args


def assemble_history(messages):
    """
    messages: AI, TOOL, AI, TOOL, etc.
    """
    query_history = ""
    n = 1
    for m in messages[1:]:  # exclude the first message
        if isinstance(m, AIMessage):
            # if there is tool call
            if hasattr(m, "tool_calls") and len(m.tool_calls) > 0:
                for tool_call in m.tool_calls:
                    tool = tool_call["name"]
                    tc_args = tool_call["args"]
                    query_history += f"Tool Call: {tool} - {tc_args}\n"
            else:
                # did not make tool calls
                query_history += f"Assistant Output {n}: {m.content}\n"
        elif isinstance(m, ToolMessage):
            query_history += f"Tool Output: {m.content}\n"
    return query_history


def generate_request_function(url):
    def process_request(query):
        import json

        import requests

        content = json.dumps({"query": query})
        print(content)
        try:
            resp = requests.post(url=url, data=content)
            ret = resp.text
            resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
        except requests.exceptions.RequestException as e:
            ret = f"An error occurred:{e}"
        print(ret)
        return ret

    return process_request


def load_func_str(tools_dir, func_str, env=None, pip_dependencies=None):
    if env is not None:
        env_list = [i.split("=") for i in env.split(",")]
        for k, v in env_list:
            print(f"set env for {func_str}: {k} = {v}")
            os.environ[k] = v

    if pip_dependencies is not None:
        import pip

        pip_list = pip_dependencies.split(",")
        for package in pip_list:
            pip.main(["install", "-q", package])
    # case 1: func is an endpoint api
    if func_str.startswith("http://") or func_str.startswith("https://"):
        return generate_request_function(func_str)

    # case 2: func is a python file + function
    elif ".py:" in func_str:
        file_path, func_name = func_str.rsplit(":", 1)
        file_path = os.path.join(tools_dir, file_path)
        file_name = os.path.basename(file_path).split(".")[0]
        spec = importlib.util.spec_from_file_location(file_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func_str = getattr(module, func_name)

    # case 3: func is a langchain tool
    elif "." not in func_str:
        return load_tools([func_str])[0]

    # case 4: func is a python loadable module
    else:
        module_path, func_name = func_str.rsplit(".", 1)
        module = importlib.import_module(module_path)
        func_str = getattr(module, func_name)
        tool_inst = func_str()
        if isinstance(tool_inst, BaseTool):
            return tool_inst
    return func_str


def load_func_args(tool_name, args_dict):
    fields = {}
    for arg_name, arg_item in args_dict.items():
        fields[arg_name] = (arg_item["type"], Field(description=arg_item["description"]))
    return create_model(f"{tool_name}Input", **fields, __base__=BaseModel)


def load_langchain_tool(tools_dir, tool_setting_tuple):
    tool_name = tool_setting_tuple[0]
    tool_setting = tool_setting_tuple[1]
    env = tool_setting["env"] if "env" in tool_setting else None
    pip_dependencies = tool_setting["pip_dependencies"] if "pip_dependencies" in tool_setting else None
    func_definition = load_func_str(tools_dir, tool_setting["callable_api"], env, pip_dependencies)
    if "args_schema" not in tool_setting or "description" not in tool_setting:
        if isinstance(func_definition, BaseTool):
            return func_definition
        else:
            raise ValueError(
                f"Tool {tool_name} is missing 'args_schema' or 'description' in the tool setting. Tool is {func_definition}"
            )
    else:
        func_inputs = load_func_args(tool_name, tool_setting["args_schema"])
        return StructuredTool(
            name=tool_name,
            description=tool_setting["description"],
            func=func_definition,
            args_schema=func_inputs,
        )


def load_yaml_tools(file_dir_path: str):
    tools_setting = yaml.safe_load(open(file_dir_path))
    tools_dir = os.path.dirname(file_dir_path)
    tools = []
    if tools_setting is None or len(tools_setting) == 0:
        return tools
    for t in tools_setting.items():
        tools.append(load_langchain_tool(tools_dir, t))
    return tools


def get_tools_descriptions(file_dir_path: str):
    # print(f"[ get_tools_descriptions ] file_dir_path: {file_dir_path}")
    tools = []
    file_path_list = []
    if os.path.isdir(file_dir_path):
        file_path_list += glob.glob(file_dir_path + "/*")
    else:
        file_path_list.append(file_dir_path)
    # print(f"[ get_tools_descriptions ] file_path_list: {file_path_list}")
    for file in file_path_list:
        # print(f"[ get_tools_descriptions ] file: {file}")
        if os.path.basename(file).endswith(".yaml"):
            tools += load_yaml_tools(file)
        else:
            pass
    return tools


def tool_renderer(tools):
    tool_strings = []
    for tool in tools:
        description = f"{tool.name} - {tool.description}"

        arg_schema = []
        for k, tool_dict in tool.args.items():
            k_type = tool_dict["type"] if "type" in tool_dict else ""
            k_desc = tool_dict["description"] if "description" in tool_dict else ""
            # arg_schema.append(f"{k} ({k_type}): {k_desc}")
            arg_schema.append({"name": k, "type": k_type, "description": k_desc, "required": True})

        tool_strings.append(f"{description}, args: {arg_schema}")
    return "\n".join(tool_strings)


def convert_json_to_tool_call(json_str):
    tool_name = json_str["tool"]
    tool_args = json_str["args"]
    tcid = str(uuid.uuid4())
    add_kw_tc = {
        "tool_calls": [
            ChatCompletionOutputToolCall(
                function=ChatCompletionOutputFunctionDefinition(arguments=tool_args, name=tool_name, description=None),
                id=tcid,
                type="function",
            )
        ]
    }
    tool_call = ToolCall(name=tool_name, args=tool_args, id=tcid)
    return add_kw_tc, tool_call


def extract_web_source(text: str):
    import re
    pattern = r"title:\s*(.+?)\s*\n.*?source:\s*(https?://[^\s]+)"
    matches = re.findall(pattern, text)
    return list(set(matches))


def cal_tokens(query: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
    tokens = tokenizer.encode(query)
    num_tokens = len(tokens)
    return num_tokens