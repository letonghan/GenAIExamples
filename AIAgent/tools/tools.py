# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import json
import requests


def send_post_request(url, payload):
    proxies = {"http": ""}
    response = requests.post(url, json=payload, proxies=proxies)
    return response


def web_search_retriever(query: str) -> str:
    """Search from web for a specific query."""
    
    tei_embedding_endpoint = os.environ.get("TEI_EMBEDDING_ENDPOINT")
    web_retriever_endpoint = os.environ.get("WEB_RETRIEVER_ENDPOINT")
    
    # prepare embedding vector
    embedding_payload = {
        "inputs": query
    }
    embedding_res = send_post_request(tei_embedding_endpoint+"/embed", embedding_payload)
    embedding_vector = embedding_res.json()[0]
    
    # searching with google
    max_retries = 5
    web_retriever_payload = {
        "text": query,
        "embedding": embedding_vector,
        "k": 8
    }
    for i in range(max_retries):
        try:
            response = send_post_request(web_retriever_endpoint+"/v1/web_retrieval", web_retriever_payload)
            if response.status_code == 200:
                # print(response)
                break
            else:
                print(f"fail to call /v1/web_retrieval, try again.")
        except Exception as e:
            print(f"Fail to call /v1/web_retrieval. Error: {e}. Tried {i+1} times.")
            
    # parse web search result
    if response:
        retrieved_docs = response.json()['retrieved_docs']
        retrieved_doc_list = [doc['text'] for doc in retrieved_docs]
        return " ".join(retrieved_doc_list)
    else:
        return ""
    
init_search = False
def rag_retriever(query: str, files: str) -> str:
    """Search from files/local database for a specific query."""
    print(query)
    print(files)

    from datetime import datetime
    start_time = datetime.now()

    dataprep_endpoint = os.environ.get("DATA_ENDPOINT")
    tei_embedding_endpoint = os.environ.get("TEI_EMBEDDING_ENDPOINT")
    retrieval_endpoint = os.environ.get("RETRIEVAL_ENDPOINT")
    tei_reranking_endpoint = os.environ.get("TEI_RERANKING_ENDPOINT")

    #############################
    # prepare data  #
    #############################
    if files:
        global init_search
        print(init_search)
        # if not init_search:
        if True:
            # need pip install qwen, pip install -e ./"[rag]"
            from qwen_agent.settings import DEFAULT_MAX_REF_TOKEN, DEFAULT_PARSER_PAGE_SIZE, DEFAULT_RAG_SEARCHERS
            from qwen_agent.tools.base import TOOL_REGISTRY, BaseTool, register_tool
            from qwen_agent.tools.doc_parser import DocParser, Record
            from qwen_agent.tools.simple_doc_parser import PARSER_SUPPORTED_FILE_TYPES

            doc_parse = DocParser({'max_ref_token': DEFAULT_MAX_REF_TOKEN, 'parser_page_size': DEFAULT_PARSER_PAGE_SIZE})

            rag_searchers = DEFAULT_RAG_SEARCHERS

            if len(rag_searchers) == 1:
                rag_search = TOOL_REGISTRY[rag_searchers[0]]({'max_ref_token': DEFAULT_MAX_REF_TOKEN})
            else:
                from qwen_agent.tools.search_tools.hybrid_search import HybridSearch
                rag_search = HybridSearch({'max_ref_token': DEFAULT_MAX_REF_TOKEN, 'rag_searchers': rag_searchers})

            init_search = True

        records = []
        for file in files:
            _record = doc_parse.call(params={'url': file})
            records.append(_record)

        if records:
            return rag_search.call(params={'query': query}, docs=[Record(**rec) for rec in records])
        else:
            return []

        """
        # headers = {'Content-Type': 'multipart/form-data'}
        headers = {}
        files_payload = {
            'link_list': (None, json.dumps(files)),
        }
        try:
            response = requests.post(dataprep_endpoint,
                headers=headers,
                files=files_payload,
                data={"chunk_size": 512, "chunk_overlap": 200})
            # response.raise_for_status()
            response = response.json()
            print(response)
        except RequestException as e:
            raise Exception(f"An unexpected error occurred: {str(e)}")
        """


    #############################
    # prepare embedding vector  #
    #############################
    embedding_payload = {
        "inputs": query
    }
    embedding_res = send_post_request(tei_embedding_endpoint+"/embed", embedding_payload)
    embedding_vector = embedding_res.json()[0]

    #############################
    # retrieval  #
    #############################

    retrieval_payload = {
        "text": query,
        "embedding": embedding_vector,
        "search_type": "similarity",
        "k": 10,
        "fetch_k": 20,
        "lambda_mult": 0.5,
    }
    print(retrieval_endpoint)
    response = requests.post(retrieval_endpoint,
        data=json.dumps(retrieval_payload),
        headers={"Content-Type": "application/json"})
    if response.ok:
        retrieved_documents = response.json()["retrieved_docs"]
        retrieved_documents = [doc["text"] for doc in retrieved_documents]
    else:
        print(f"Request for retrieval failed due to {response.text}.")
        retrieved_documents = []

    # print(retrieved_documents)

    #############################
    # rerank the retrieved docs #
    #############################
    rerank_payload = {
        "query": query,
        "texts": retrieved_documents
    }
    rerank_res = send_post_request(tei_reranking_endpoint+"/rerank", rerank_payload).json()
    highest_score_index = sorted(rerank_res, key=lambda x: x['score'])
    # print(highest_score_index)

    # final_result = retrieved_documents[highest_score_index]
    final_result = "\n".join(retrieved_documents[:5]) + '\n' + \
        "The relevant content has already been retrieved and updated in the previous system message."
    # print("================")
    # print(final_result)

    elapsed_time = datetime.now() - start_time
    print("==================")
    print(f"Execution time: {elapsed_time}")
    # exit()
    return final_result


def image_generation(prompt: str) -> str:
    image_gen_endpoint = os.environ.get("IMAGE_GEN_ENDPOINT")
    image_payload = {
        "prompt": prompt,
        "num_images_per_prompt":1
    }
    max_retries = 5
    
    for i in range(max_retries):
        try:
            response = send_post_request(image_gen_endpoint+"/v1/text2image", image_payload)
            if response.status_code == 200:
                break
            else:
                print(f"fail to call /v1/text2image, try again.")
        except Exception as e:
            print(f"Fail to call /v1/text2image. Error: {e}. Tried {i+1} times.")
            
    if response:
        image_list = response.json()['images']
        return image_list[0]
    else:
        return ""

