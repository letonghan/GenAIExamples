# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

WORKPATH=$(dirname "$PWD")/..
echo "WORKDIR=${WORKDIR}"
export ip_address=$(hostname -I | awk '{print $1}')
export no_proxy=${ip_address}
export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export model="Qwen/Qwen2.5-72B-Instruct"
export LOGFLAG="true"


function start_tgi_ragagent() {
    export HF_CACHE_DIR="/home/kding1/letong/data"
    export NUM_SHARDS=2

    docker compose -f compose.yaml up -d
}

start_tgi_ragagent


function start_agent_planner() {
    export tool_yaml_path="/home/kding1/letong/ai_agent/tools/agent_tools.yaml"
    export llm_endpoint_url="http://${ip_address}:8085"
    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:3001"
    export WEB_RETRIEVER_ENDPOINT="http://${ip_address}:3003"
    export TEI_RERANKING_ENDPOINT="http://${ip_address}:3004"

    python agent_planner.py
}

start_agent_planner