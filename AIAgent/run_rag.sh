export ip_address=$(hostname -I | awk '{print $1}')
export no_proxy=${no_proxy}
export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export recursion_limit_worker=12
export llm_endpoint_url="http://10.7.4.144:8085"
export model="Qwen/Qwen2.5-72B-Instruct"
export LOGFLAG="true"
export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:3001"
export TEI_RERANKING_ENDPOINT="http://${ip_address}:3004"
export DATA_ENDPOINT="http://${ip_address}:6007/v1/dataprep"
export RETRIEVAL_ENDPOINT="http://${ip_address}:7000/v1/retrieval"

export PYTHONPATH="/home/kaokaolv/lkk/agent_demo/GenAIComps"
export tool_yaml_path="tools/agent_rag_tools.yaml"

# python ai_agent.py
python agent_planner.py
