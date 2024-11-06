from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_cohere import ChatCohere
from typing import TypedDict, Annotated
from agent_tools import find_relevant_diffs
from dotenv import load_dotenv
from contextlib import ExitStack
import operator
import uuid

load_dotenv()

index_name = "rootly"
model = ChatCohere(model='command-r-plus')
search_tool = TavilySearchResults(max_results=4) 

stack = ExitStack()
memory = stack.enter_context(SqliteSaver.from_conn_string(":memory:"))

agent_prompt = '''
You are a professional software engineer that can read code and understand the changes made in a PR.
You will be given an issue description and you need to find the PR that is most relevant to the issue description.
You need to output which PR is most relevant, and a summary of the changes made in the PR, with some recommendations for fixing the issue.

When trying to find a way to resolve the issue, think step by step about how you would fix the issue if you were a human developer.
Include the PR URL andline numbers for suggestions if possible.
 
If absolutely necessary, you can perform web searches for more information on how to solve the issue.
'''


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_llm(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
    

messages = [HumanMessage(content="Broken pinecone upsert")]
user_uuid = str(uuid.uuid4())
print(user_uuid)

agent = Agent(model, tools=[find_relevant_diffs, search_tool], system=agent_prompt, checkpointer=memory)
thread = {"configurable": {"thread_id": user_uuid}}

output = []

for event in agent.graph.stream({"messages": messages}, thread):
    for v in event.values():
        output.append(v)

print(output[-1]['messages'][-1].content)