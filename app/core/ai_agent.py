from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

def get_response_from_ai_agents(llm_id, query, allow_search, system_prompt):
    llm = ChatGroq(model=llm_id)
    tools = [TavilySearch(max_results=2)] if allow_search else []
    agent = create_react_agent(model=llm, tools=tools)
    messages_with_prompt = [system_prompt] + query
    state = {"messages": messages_with_prompt}
    response = agent.invoke(state)

    messages = response.get("messages", [])
    ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]

    # Return the last AI message
    return ai_messages[-1] if ai_messages else "No response"
