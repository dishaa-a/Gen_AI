from langchain_core.prompts import PromptTemplate

#template
template = PromptTemplate(
    template = 
"""You are a research assistant. Your task is to summarize the research paper titled "{paper_title}" in a {summary_style} style and {summary_length} length. Please provide a clear and concise summary of the key points, findings, and contributions of the paper.
Summary:
""",
input_variables = ["paper_title", "summary_style", "summary_length"]
)

template.save("research_summary_template.json")