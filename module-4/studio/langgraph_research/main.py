from graphs.research_graph import create_research_graph

def main():
    """Create the research graph and make it available for use"""
    
    # Create the research graph
    graph = create_research_graph()
    
    # Example usage
    # config = {"topic": "AI Safety", "max_analysts": 3}
    # for event in graph.stream(config):
    #     if event["type"] == "human-feedback":
    #         # Handle human feedback here
    #         feedback = get_human_feedback()  # Your function to get feedback
    #         yield {"human_analyst_feedback": feedback}
    
    return graph

if __name__ == "__main__":
    main()