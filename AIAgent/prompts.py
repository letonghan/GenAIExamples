# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from langchain import PromptTemplate

# Create initial tasks using plan and solve prompting
# https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting
start_goal_prompt = PromptTemplate(
    template="""You are a task creation AI called AIAgent. 
You answer in the "{language}" language. You have the following objective "{goal}". 
Return a list of search queries that would be required to answer the entirety of the objective. 
Limit the list to a maximum of 5 queries. Ensure the queries are as succinct as possible. 
For simple questions use a single query.

Return the response as a JSON array of strings. Examples:

query: "Who is considered the best NBA player in the current season?", answer: ["current NBA MVP candidates"]
query: "How does the Olympicpayroll brand currently stand in the market, and what are its prospects and strategies for expansion in NJ, NY, and PA?", answer: ["Olympicpayroll brand comprehensive analysis 2023", "customer reviews of Olympicpayroll.com", "Olympicpayroll market position analysis", "payroll industry trends forecast 2023-2025", "payroll services expansion strategies in NJ, NY, PA"]
query: "How can I create a function to add weight to edges in a digraph using {language}?", answer: ["algorithm to add weight to digraph edge in {language}"]
query: "What is the current weather in New York?", answer: ["current weather in New York"]
query: "5 + 5?", answer: ["Sum of 5 and 5"]
query: "What is a good homemade recipe for KFC-style chicken?", answer: ["KFC style chicken recipe at home"]
query: "What are the nutritional values of almond milk and soy milk?", answer: ["nutritional information of almond milk", "nutritional information of soy milk"]
""",
    input_variables=["goal", "language"],
)


start_goal_prompt_zh = PromptTemplate(
    template="""你是一款名为 AIAgent 的任务生成 AI。请使用中文回答。你的目标是“{goal}”。

返回回答此目标所需的搜索查询列表，最多包含 5 个查询，并确保查询尽量简洁。如果是简单问题，则使用单一查询。

将答案以 JSON 字符串数组的格式返回。示例如下：

query: "当前赛季被认为最好的 NBA 球员是谁？", answer: ["当前 NBA MVP 候选人"] 
query: "Olympicpayroll 品牌目前在市场中的地位如何？其在 NJ、NY 和 PA 的扩展前景和策略是什么？", answer: ["Olympicpayroll 品牌 2023 综合分析", "Olympicpayroll.com 的用户评价", "Olympicpayroll 市场地位分析", "2023-2025 年工资单行业趋势预测", "在 NJ、NY、PA 扩展的工资服务策略"] 
query: "杏仁奶和豆奶的营养价值是什么？", answer: ["杏仁奶的营养信息", "豆奶的营养信息"]
""",
    input_variables=["goal"],
)


summarize_prompt = PromptTemplate(
    template="""You must answer in the "{language}" language.

    Combine the following text into a cohesive document:

    "{text}"

    Write using clear markdown formatting in a style expected of the goal "{goal}".
    Be as clear, informative, and descriptive as necessary.
    You will not make up information or add any information outside of the above text.
    Only use the given information and nothing more.

    If there is no information provided, say "There is nothing to summarize".
    """,
    input_variables=["goal", "language", "text"],
)


summarize_prompt_zh = PromptTemplate(
    template="""请使用中文回答。

请将以下文本整合成连贯的文档：

“{text}”

请使用清晰的 Markdown 格式，并确保符合“{goal}”的写作风格要求。内容应尽可能清晰、信息丰富且描述详尽。
不要添加任何虚构信息或额外内容，仅使用所给信息。

如果没有提供任何信息，请回答“没有可以总结的内容”。
    """,
    input_variables=["goal", "text"],
)


REACT_AGENT_LLAMA_PROMPT = """\
Given the user request, think through the problem step by step.
Observe the outputs from the tools in the execution history, and think if you can come up with an answer or not. If yes, provide the answer. If not, make tool calls.
When you cannot get the answer at first, do not give up. Reflect on the steps you have taken so far and try to solve the problem in a different way.

You have access to the following tools:
{tools}

Begin Execution History:
{history}
End Execution History.

If you need to call tools, use the following format:
{{"tool":"tool 1", "args":{{"input 1": "input 1 value", "input 2": "input 2 value"}}}}
{{"tool":"tool 2", "args":{{"input 3": "input 3 value", "input 4": "input 4 value"}}}}
Multiple tools can be called in a single step, but always separate each tool call with a newline.

IMPORTANT: You MUST ALWAYS make tool calls unless you can provide an answer. Make each tool call in JSON format in a new line.

If you can generate an answer, provide the answer in the following format in a new line:
{{"answer": "your answer here"}}

Follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer “invalid question”.
2. If you are uncertain or do not know the answer, answer “I don't know”.
3. Give concise, factual and relevant answers.

User request: {input}
Now begin!
"""
