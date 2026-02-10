from crewai import Agent, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool


def check_local_knowledge(query, context, llm = None):
    prompt = '''Role : Question- Answering Assistant
    Task : Determine whether the system can answer the user's querstion based on the proivided text.
    Output Format : Answer : Yes/ No
    ...
    User Question : {query}
    Text : {text}'''
    
    formatted_prompt = prompt.format(text=context, query = query)
    response = llm.invoke(formatted_prompt)
    return response.content.strip().lower() == 'yes'

def setup_web_scarapign_agent(llm=None):
    search_tool = SerperDevTool()
    scrape_website = ScrapeWebsiteTool()
    web_search_agent = Agent(
        role = 'Expert Web Search Agent',
        goal = 'Indentify and  retrieve relevant web data for users queries',
        tools = [search_tool],
        llm = llm,
        verbose = True 
    )

    web_scarber_agent = Agent(
        role = 'Expert Web Scraber Agent',
        goal = 'Extract and analyze content from web pages',
        tools = [scrape_website],
        llm = llm , 
        verbose = True 
    )

    crew  = Crew(
        agents = [web_search_agent , web_scarber_agent],
        verbose= True 
    )
    return crew 


def get_web_content(query): 
    crew = setup_web_scarapign_agent()
    result = crew.kickoff(inputs={'topics' : query})
    return result.raw 
