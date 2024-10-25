# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import requests


def send_post_request(url, payload):
    proxies = {"http": ""}
    response = requests.post(url, json=payload, proxies=proxies)
    return response


def web_search_retriever(query: str) -> str:
    """Search from web for a specific query."""
    from datetime import datetime

    start_time = datetime.now()
    
    tei_embedding_endpoint = os.environ.get("TEI_EMBEDDING_ENDPOINT")
    web_retriever_endpoint = os.environ.get("WEB_RETRIEVER_ENDPOINT")
    tei_reranking_endpoint = os.environ.get("TEI_RERANKING_ENDPOINT")
    
    #############################
    # prepare embedding vector  #
    #############################
    embedding_payload = {
        "inputs": query
    }
    embedding_res = send_post_request(tei_embedding_endpoint+"/embed", embedding_payload)
    embedding_vector = embedding_res.json()[0]
    
    #############################
    #   searching with google   #
    #############################
    web_retriever_payload = {
        "text": query,
        "embedding": embedding_vector
    }
    response = send_post_request(web_retriever_endpoint+"/v1/web_retrieval", web_retriever_payload)
    print(response)
    retrieved_docs = response.json()['retrieved_docs']
    retrieved_doc_list = [doc['text'] for doc in retrieved_docs]
    
    #############################
    # rerank the retrieved docs #
    #############################
    rerank_payload = {
        "query": query,
        "texts": retrieved_doc_list
    }
    rerank_res = send_post_request(tei_reranking_endpoint+"/rerank", rerank_payload).json()
    highest_score_index = max(rerank_res, key=lambda x: x['score'])['index']
    
    final_result = retrieved_doc_list[highest_score_index]
    print("================")
    print(final_result)
    
    elapsed_time = datetime.now() - start_time
    print("==================")
    print(f"Execution time: {elapsed_time}")
    return final_result

