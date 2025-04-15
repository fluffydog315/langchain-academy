from langgraph.constants import START, END
from langgraph.graph import StateGraph

from models.schemas import ResearchGraphState
from nodes.analyst_nodes import create_analysts, human_feedback, initiate_all_interviews
from nodes.report_nodes import write_report, write_introduction, write_conclusion, finalize_report
from graphs.interview_graph import create_interview_graph

def create_research_graph():
    """Create and return the research graph"""
    
    # Add nodes and edges 
    builder = StateGraph(ResearchGraphState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("conduct_interview", create_interview_graph())
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)

    # Logic
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges(
        "human_feedback", 
        initiate_all_interviews, 
        ["create_analysts", "conduct_interview"]
    )
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(
        ["write_conclusion", "write_report", "write_introduction"], 
        "finalize_report"
    )
    builder.add_edge("finalize_report", END)
    
    return builder.compile(interrupt_before=['human_feedback'])