from langgraph.constants import START, END
from langgraph.graph import StateGraph

from models.schemas import InterviewState
from nodes.interview_nodes import (
    generate_question,
    search_web,
    search_wikipedia,
    generate_answer,
    save_interview,
    route_messages,
    write_section
)

def create_interview_graph():
    """Create and return the interview graph"""
    
    # Add nodes and edges 
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("search_web", search_web)
    interview_builder.add_node("search_wikipedia", search_wikipedia)
    interview_builder.add_node("answer_question", generate_answer)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_section", write_section)

    # Flow
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_web")
    interview_builder.add_edge("ask_question", "search_wikipedia")
    interview_builder.add_edge("search_web", "answer_question")
    interview_builder.add_edge("search_wikipedia", "answer_question")
    interview_builder.add_conditional_edges(
        "answer_question", 
        route_messages,
        ['ask_question', 'save_interview']
    )
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)
    
    return interview_builder.compile()