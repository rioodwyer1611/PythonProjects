import asyncio
from url_dev_sdk import run_workflow, WorkflowInput
from seek_scraper import get_seek_listings
from indeed_scraper import get_indeed_listings

async def run_url_agent(user_query: str):
    result = await run_workflow(WorkflowInput(input_as_text=user_query))
    final = []
    for url in result['seek_url_list']:
        temp = get_seek_listings(url, debug=True)
        final.append(temp)
    for url in result['indeed_url_list']:
        temp = get_indeed_listings(url, debug=True)
        final.append(temp)
    return final
