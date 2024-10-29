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
from utils.utils import extract_task_list, get_args
from prompts import start_goal_prompt, start_goal_prompt_zh, summarize_prompt, summarize_prompt_zh
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
        max_new_tokens=128,
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

    return task_list


@register_microservice(
    name="opea_service@ai_agent",
    endpoint="/v1/agent/execute",
    host="0.0.0.0",
    port=service_port,
)
@register_statistics(names=["opea_service@ai_agent"])
async def agent_start(input: LLMParamsDoc):
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
async def agent_start(input: AgentSumDoc):
    if logflag:
        logger.info("[ Summarize ] calling /v1/agent/start router")
        logger.info(input)

    goal = input.goal
    language = input.language
    results = input.results

    text = " ".join(results)
    prompt = summarize_prompt.format(goal=goal, language=language, text=text)

    if logflag:
        logger.info(f"[ Summarize ] final input prompt: {prompt}")
    
    text_generation = await llm.text_generation(
        prompt=prompt,
        stream=input.streaming,
        max_new_tokens=input.max_tokens,
        repetition_penalty=input.repetition_penalty,
        temperature=input.temperature,
        top_k=input.top_k,
        top_p=input.top_p,
    )
    
    if input.streaming:
        async def stream_generator():
            chat_response = ""
            async for text in text_generation:
                chat_response += text
                chunk_repr = text
                if logflag:
                    logger.info(f"[ Summarize ] chunk:{chunk_repr}")
                yield f"data: {chunk_repr}\n\n"
            if logflag:
                logger.info(f"[ Summarize ] stream response: {chat_response}")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    else:
        if logflag:
            logger.info(f"[ Summarize ] text generation: {text_generation}")
        return text_generation


if __name__ == "__main__":
    opea_microservices["opea_service@ai_agent"].start()

