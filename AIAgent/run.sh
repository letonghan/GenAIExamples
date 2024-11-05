# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


export ip_address=$(hostname -I | awk '{print $1}')
export no_proxy=${ip_address}
export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export recursion_limit_worker=12
export llm_endpoint_url="http://${ip_address}:8085"
export model="Qwen/Qwen2.5-32B-Instruct"
export LOGFLAG="true"
export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:3001"
export WEB_RETRIEVER_ENDPOINT="http://${ip_address}:3003"
export IMAGE_GEN_ENDPOINT="http://${ip_address}:3005"
export PYTHONPATH="/home/kding1/letong/GenAIComps/"
export tool_yaml_path="/home/kding1/letong/agent_lkk/AIAgent/tools/agent_tools.yaml"


function start_tgi_ragagent() {
    export HF_CACHE_DIR="/home/kding1/letong/data"
    export HABANA_VISIBLE_DEVICES="0,1,2,3"
    export NUM_SHARDS=4

    docker compose -f compose.yaml up -d
}

start_tgi_ragagent

python ai_agent.py

