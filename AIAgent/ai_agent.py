# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from huggingface_hub import AsyncInferenceClient

from comps import (
    CustomLogger,
    LLMParamsDoc,
    AgentTaskDoc,
    AgentSumDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
)
from utils import extract_task_list
from prompts import start_goal_prompt, analyze_task_prompt, summarize_prompt


logger = CustomLogger("ai_agent")
logflag = os.getenv("LOGFLAG", False)

llm_endpoint = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")
llm = AsyncInferenceClient(
    model=llm_endpoint,
    timeout=600,
)


@register_microservice(
    name="opea_service@ai_agent",
    service_type=ServiceType.LLM,
    endpoint="/v1/agent/start",
    host="0.0.0.0",
    port=7071,
)
@register_statistics(names=["opea_service@ai_agent"])
async def agent_start(input: LLMParamsDoc):
    if logflag:
        logger.info("[ Start ] calling /v1/agent/start router")
        logger.info(input)

    goal = input.query
    # set Chinese as default language
    language = "Chinese"

    # use pre-defined prompt for llm inference
    prompt = start_goal_prompt.format(goal=goal, language=language)

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

    return task_list


@register_microservice(
    name="opea_service@ai_agent",
    endpoint="/v1/agent/execute",
    host="0.0.0.0",
    port=7071,
)
@register_statistics(names=["opea_service@ai_agent"])
async def agent_execute(input: AgentTaskDoc):
    if logflag:
        logger.info("[ Execute ] calling /v1/agent/execute router")
        logger.info(input)

    goal = input.goal
    task = input.task
    # set Chinese as default language
    language = "Chinese"

    # use pre-defined prompt for llm inference
    prompt = analyze_task_prompt.format(goal=goal, task=task, language=language)

    if logflag:
        logger.info(f"[ Execute ] final input prompt: {prompt}")
    
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
        logger.info(f"[ Execute ] text generation: {text_generation}")

    return text_generation


@register_microservice(
    name="opea_service@ai_agent",
    service_type=ServiceType.LLM,
    endpoint="/v1/agent/summarize",
    host="0.0.0.0",
    port=7071,
)
@register_statistics(names=["opea_service@ai_agent"])
async def agent_start(input: AgentSumDoc):
    if logflag:
        logger.info("[ Summarize ] calling /v1/agent/start router")
        logger.info(input)

    goal = input.goal
    # set Chinese as default language
    language = input.language
    results = input.results

    text = " ".join(results)

    # use pre-defined prompt for llm inference
    prompt = summarize_prompt.format(goal=goal, language=language, text=text)

    if logflag:
        logger.info(f"[ Summarize ] final input prompt: {prompt}")
    
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
        logger.info(f"[ Summarize ] text generation: {text_generation}")

    return text_generation


if __name__ == "__main__":
    opea_microservices["opea_service@ai_agent"].start()
