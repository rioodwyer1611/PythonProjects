import asyncio
from url_dev_sdk import run_workflow, WorkflowInput
from seek_scraper import get_seek_listings
from indeed_scraper import get_indeed_listings

def run_url_agent(user_query: str):
    result = asyncio.run(run_workflow(WorkflowInput(input_as_text=user_query)))
    print(result)
    seek_results = []
    indeed_results = []
    for url in result["seek_url_list"]:
        seek_results += get_seek_listings(url)
    for url in result["indeed_url_list"]:
        indeed_results += get_indeed_listings(url)
    final = seek_results + indeed_results
    print(final)
    return final

def remove_duplicate_entries(job_listings: dict[str]):
    pass
