from toolbox import Retriever
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_cohere import ChatCohere
from langchain_core.messages import SystemMessage, HumanMessage
import os

load_dotenv()
retriever = Retriever()

@tool('find_relevant_diffs')
def find_relevant_diffs(issue_description: str) -> str:
    '''
    gets the diffs from the PRs most likely to have caused the issue
    issue_description: str - the description of the issue
    '''
    prs = []

    search_results = retriever.semantic_search(issue_description, index_name='rootly', top_k=5)
    for result in search_results['matches']:
        prs.append({"id": result['id'], "url": result['metadata']['url']})

    ## grab the diffs for the IDs found from semantic search
    pr_ids = [pr['id'] for pr in prs]
    diffs = retriever.get_diffs(owner='ethanbailie', repo='agentic_code_observer', pr_ids=pr_ids)

    ## make a dictionary with the PR id as the key and the url and diff as the value
    diffs_with_metadata = {pr['id']: {"url": pr['url'], "diff": diffs[pr['id']]} for pr in prs}

    chat = ChatCohere(model='command-r-plus')
    messages = [
        SystemMessage(content='''
        You are a professional software engineer that can read code and understand the changes made in a PR.
        You will be given a list of PRs and their diffs, and an issue description.
        Your job is to find the PR that is most relevant to the issue description.
        You will output the PR id and url of the most relevant PR with a description of the changes made in the PR.
        Do not suggest any fixes, just the PR id and the description of the changes found within.
        For the description of changes, denote which lines each significant change is on.
        '''),
        HumanMessage(content=f'''
        Here are the PRs most likely to have caused the issue, in the form of a dictionary with the PR id as the key and the diff as the value:
        {diffs_with_metadata}
        
        Which code PR is most relevant to the issue description: {issue_description}?
        ''')
    ]
    
    response = chat.invoke(messages)
    return {"messages": [response.content]}
