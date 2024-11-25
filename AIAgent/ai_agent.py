# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from huggingface_hub import AsyncInferenceClient

from comps import (
    CustomLogger,
    LLMParamsDoc,
    AgentSumDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
)
from utils.utils import extract_task_list, get_args, cal_tokens
from prompts import start_goal_prompt, summarize_prompt, summarize_prompt_new
from agent_planner import AgentPlanner
from fastapi.responses import StreamingResponse


logger = CustomLogger("ai_agent")
logflag = os.getenv("LOGFLAG", False)
service_port = os.getenv("PORT", 7071)
args, _ = get_args()
planner = AgentPlanner(args)

llm_endpoint = os.getenv("llm_endpoint_url", "http://localhost:8080")
llm = AsyncInferenceClient(
    model=llm_endpoint,
    timeout=600,
)


@register_microservice(
    name="opea_service@ai_agent",
    service_type=ServiceType.LLM,
    endpoint="/v1/agent/start",
    host="0.0.0.0",
    port=service_port,
)
@register_statistics(names=["opea_service@ai_agent"])
async def agent_start(input: LLMParamsDoc):
    if logflag:
        logger.info("[ Start ] calling /v1/agent/start router")
        logger.info(input)

    goal = input.query

    # use pre-defined prompt for llm inference
    prompt = start_goal_prompt.format(goal=goal, language="Chinese")

    if logflag:
        logger.info(f"[ Start ] final input prompt: {prompt}")
    
    text_generation = await llm.text_generation(
        prompt=prompt,
        stream=False,
        max_new_tokens=input.max_tokens,
        repetition_penalty=input.repetition_penalty,
        temperature=input.temperature,
        top_k=input.top_k,
        top_p=input.top_p,
    )

    if logflag:
        logger.info(f"[ Start ] text generation: {text_generation}")
    
    task_list = extract_task_list(text_generation)

    if logflag:
        logger.info(f"[ Start ] task list: {task_list}")

    return task_list[:5]


@register_microservice(
    name="opea_service@ai_agent",
    endpoint="/v1/agent/execute",
    host="0.0.0.0",
    port=service_port,
)
@register_statistics(names=["opea_service@ai_agent"])
async def agent_execute(input: LLMParamsDoc):
    if logflag:
        logger.info("[ Execute ] calling /v1/agent/execute router")
        logger.info(input)

    input_query = input.query
    config = {"recursion_limit": args.recursion_limit}
    
    generator = planner.stream_generator(input_query, config)
    
    return StreamingResponse(generator, media_type="text/event-stream")


@register_microservice(
    name="opea_service@ai_agent",
    service_type=ServiceType.LLM,
    endpoint="/v1/agent/summarize",
    host="0.0.0.0",
    port=service_port,
)
@register_statistics(names=["opea_service@ai_agent"])
async def agent_summarize(input: AgentSumDoc):
    if logflag:
        logger.info("[ Summarize ] calling /v1/agent/start router")
        logger.info(input)

    goal = input.goal
    language = input.language
    results = input.results

    text = " ".join(results)
    prompt = summarize_prompt_new.format(goal=goal, language=language, text=text)

    if logflag:
        logger.info(f"[ Summarize ] final input prompt: {prompt}")
    
    framework = "vllm"

    if framework == "vllm":
        from langchain_community.llms import VLLMOpenAI
        llm_endpoint = os.getenv("vLLM_ENDPOINT", "http://localhost:8086")
        model_name = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")
        vllm = VLLMOpenAI(openai_api_key="EMPTY", openai_api_base=llm_endpoint + "/v1", model_name=model_name)
        parameters = {
            "max_tokens": input.max_tokens,
            "top_p": input.top_p,
            "temperature": input.temperature,
            "frequency_penalty": input.frequency_penalty,
            "presence_penalty": input.presence_penalty,
        }
        if logflag:
            logger.info(f"params: {parameters}")
        
        if input.streaming:
            async def stream_generator():
                chat_response = ""
                async for text in vllm.astream(prompt, **parameters):
                    chat_response += text
                    if logflag:
                        logger.info(f"[ Summarize - vllm ] chunk: {text}")
                    lines = text.split("\n")
                    chunk_repr = "\\n".join(lines)
                    chunk_repr = chunk_repr.replace(" ", "&nbsp;")
                    if chunk_repr == "<|im_end|>":
                        break
                    yield f"data: {chunk_repr}\n\n"
                if logflag:
                    logger.info(f"[ Summarize - vllm ] stream response: {chat_response}")
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            response = await vllm.ainvoke(prompt, **parameters)
            if logflag:
                logger.info(f"[ Summarize - vllm ] text generation: {response}")

            return response
    
    elif framework == "tgi":
        text_generation = await llm.text_generation(
            prompt=prompt,
            stream=input.streaming,
            max_new_tokens=input.max_tokens,
            repetition_penalty=input.repetition_penalty,
            frequency_penalty=input.frequency_penalty,
            temperature=input.temperature,
            top_k=input.top_k,
            top_p=input.top_p,
        )
        
        if input.streaming:
            async def stream_generator():
                chat_response = ""
                async for text in text_generation:
                    chat_response += text
                    lines = text.split("\n")
                    chunk_repr = "\\n".join(lines)
                    chunk_repr = chunk_repr.replace(" ", "&nbsp;")
                    if logflag:
                        logger.info(f"[ Summarize - tgi ] chunk:{chunk_repr}")
                    if chunk_repr == "<|im_end|>":
                        break
                    yield f"data: {chunk_repr}\n\n"
                if logflag:
                    logger.info(f"[ Summarize - tgi ] stream response: {chat_response}")
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            if logflag:
                logger.info(f"[ Summarize - tgi ] text generation: {text_generation}")
            return text_generation



if __name__ == "__main__":
    opea_microservices["opea_service@ai_agent"].start()

