from crewai import Agent, Crew, Task
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

def setup_web_scarapign_agent(llm=None, query=''):
    search_tool = SerperDevTool()
    scrape_website = ScrapeWebsiteTool()
    web_search_agent = Agent(
        role = 'Expert Web Search Agent',
        goal = 'Indentify and  retrieve relevant web data for users queries',
        backstory = 'You are an experienced web search specialist with expertise in finding accurate and relevant information online.',
        tools = [search_tool],
        llm = llm,
        verbose = True 
    )

    web_scarber_agent = Agent(
        role = 'Expert Web Scraber Agent',
        goal = 'Extract and analyze content from web pages',
        backstory = 'You are a skilled web scraping expert who can extract and analyze content from various web sources.',
        tools = [scrape_website],
        llm = llm , 
        verbose = True 
    )

    search_task = Task(
        description = f'Search the web for information about: {query}. Find the most relevant and reliable sources.',
        expected_output = 'A list of relevant URLs and brief descriptions of the content found.',
        agent = web_search_agent
    )

    scrape_task = Task(
        description = f'Scrape and extract detailed information from the web pages found for the query: {query}',
        expected_output = 'Comprehensive and relevant information extracted from the web pages that answers the query.',
        agent = web_scarber_agent
    )

    crew  = Crew(
        agents = [web_search_agent , web_scarber_agent],
        tasks = [search_task, scrape_task],
        verbose= True 
    )
    return crew 


def get_web_content(query, llm=None): 
    crew = setup_web_scarapign_agent(llm=llm, query=query)
    result = crew.kickoff()
    return result.raw 
