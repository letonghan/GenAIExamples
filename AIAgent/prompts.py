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


summarize_prompt = PromptTemplate(
    template="""You must answer in the "{language}" language.

    Combine the following text into a cohesive document:

    "{text}"

    Write using clear format in a style expected of the goal "{goal}".
    Be as clear, informative, and descriptive as necessary.
    You will not make up information or add any information outside of the above text.
    Only use the given information and nothing more.

    If there is no information provided, say "There is nothing to summarize".
    """,
    input_variables=["goal", "language", "text"],
)


summarize_prompt_new = PromptTemplate(
    template="""You must respond in the "{language}" language.

Task: Summarize and structure the provided text into a cohesive document using clear Markdown formatting, adhering strictly to the following rules, ensuring no repetitive phrases or sentences.

Output Format:
You must use clear and valid Markdown formatting for your response.
Ensure proper structure, including headers, bullet points, numbered lists, or other Markdown elements as appropriate to achieve readability and clarity.

Content Rules:
Write in a style aligned with the specified goal: "{goal}".
Use only the information provided in input text and nothing else. Do not make up, infer, or add details not explicitly mentioned.
If no information is provided, state: "There is nothing to summarize" and end your response.

Clarity and Relevance:
Your output must be informative, precise, and descriptive, staying true to the text's content and purpose.
Avoid redundancy and focus on presenting the key points in a logical and concise manner.

Input text:
{text}

Proceed to create a well-organized Markdown document based on the above instructions. Stop when the main points are covered.
    """,
    input_variables=["language", "goal", "text"]
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

If you can generate an answer, provide it in a thorough, step-by-step manner. Explain the answer with as much detail and depth as possible, covering all relevant points and perspectives. Always provide the answer in {language} language.
Reflect on the problem and consider if there are additional steps or explanations that can further elaborate the answer. If possible, include examples, comparisons, or expanded explanations to ensure comprehensive coverage.
Structure your answer with sub-points or sections if applicable, addressing each aspect of the request in detail.

User request: {input}
Now begin!
"""
