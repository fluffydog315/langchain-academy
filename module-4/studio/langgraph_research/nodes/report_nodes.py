from langchain_core.messages import SystemMessage, HumanMessage

from config import llm, report_writer_instructions, intro_conclusion_instructions
from models.schemas import ResearchGraphState

def write_report(state: ResearchGraphState):
    """ Node to write the final report body """

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
    report = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Write a report based upon these memos.")
    ]) 
    return {"content": report.content}

def write_introduction(state: ResearchGraphState):
    """ Node to write the introduction """

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    instructions = intro_conclusion_instructions.format(
        topic=topic, 
        formatted_str_sections=formatted_str_sections
    )    
    intro = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=f"Write the report introduction")
    ]) 
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    """ Node to write the conclusion """

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    instructions = intro_conclusion_instructions.format(
        topic=topic, 
        formatted_str_sections=formatted_str_sections
    )    
    conclusion = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=f"Write the report conclusion")
    ]) 
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, 
    and reflect on them to write the intro/conclusion """

    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}