from langchain_core.messages import SystemMessage, HumanMessage

from config import llm, analyst_instructions
from models.schemas import GenerateAnalystsState, Perspectives
from utils.error_handler import record_error, is_error_state, clear_error

def create_analysts(state: GenerateAnalystsState):
    """ Create analysts """
    try:
        topic = state['topic']
        max_analysts = state['max_analysts']
        human_analyst_feedback = state.get('human_analyst_feedback', '')
            
        # Enforce structured output
        structured_llm = llm.with_structured_output(Perspectives)

        # System message
        system_message = analyst_instructions.format(
            topic=topic,
            human_analyst_feedback=human_analyst_feedback, 
            max_analysts=max_analysts
        )

        # Generate question 
        analysts = structured_llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content="Generate the set of analysts.")
        ])
        
        # Write the list of analysis to state
        return {"analysts": analysts.analysts}
    except Exception as e:
        # Use record_error but maintain original return type
        error_update = record_error(
            state=state,
            error=e,
            node_name="create_analysts"
        )
        # Return with correct type
        return error_update

def human_feedback(state: GenerateAnalystsState):
    """ No-op node that should be interrupted on """
    pass

def initiate_all_interviews(state: dict):
    """ Conditional edge to initiate all interviews via Send() API or return to create_analysts """    
    from langgraph.constants import Send
    from langchain_core.messages import HumanMessage

    # Check if human feedback
    human_analyst_feedback = state.get('human_analyst_feedback', 'good')
    if human_analyst_feedback.lower() != 'good':
        # Return to create_analysts
        return "create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        topic = state["topic"]
        return [
            Send("conduct_interview", {
                "analyst": analyst,
                "messages": [HumanMessage(
                    content=f"So you said you were writing an article on {topic}?"
                )]
            }) for analyst in state["analysts"]
        ]