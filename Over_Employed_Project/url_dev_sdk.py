from pydantic import BaseModel
from agents import Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace

class SeekUrlCreationSchema__SeekUrlList(BaseModel):
  url1: str
  url2: str
  url3: str
  url4: str
  url5: str


class SeekUrlCreationSchema(BaseModel):
  seek_url_list: SeekUrlCreationSchema__SeekUrlList


class IndeedUrlCreationSchema__IndeedUrlList(BaseModel):
  indeed_url1: str
  indeed_url2: str
  indeed_url3: str
  indeed_url4: str
  indeed_url5: str


class IndeedUrlCreationSchema__SeekUrlList(BaseModel):
  seek_url1: str
  seek_url2: str
  seek_url3: str
  seek_url4: str
  seek_url5: str


class IndeedUrlCreationSchema(BaseModel):
  indeed_url_list: IndeedUrlCreationSchema__IndeedUrlList
  seek_url_list: IndeedUrlCreationSchema__SeekUrlList


seek_url_creation = Agent(
  name="SEEK Url Creation",
  instructions="""You are a helpful assistant skilled in understanding basic english and human language patterns, you are to interpret the inputs given to you. You will, using the job type, field, example companies, pay, and location to generate a list of urls applicable to the given criteria. You will create a list of several urls, to hit any and all possible job listings which the user may find suitable.

SEEK URLs are in the form of:
https://www.seek.com.au/job-type-jobs/in-location?workarrangement=2%2C1%2C3&worktype=245%2C244%2C243%2C242

Examples:
https://www.seek.com.au/supplements-jobs-in-sport-recreation/in-All-Brisbane-QLD/on-site?daterange=31&salaryrange=35-60&salarytype=hourly&worktype=243%2C245

https://www.seek.com.au/analytics-jobs/in-All-Brisbane-QLD/hybrid?daterange=3&salaryrange=150000-&salarytype=annual&worktype=244%2C242

https://www.seek.com.au/jobs/in-Southern-Suburbs-&-Logan-Brisbane-QLD/part-time/remote?daterange=7&salaryrange=5000-30000&salarytype=monthly
""",
  model="gpt-4.1",
  output_type=SeekUrlCreationSchema,
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


indeed_url_creation = Agent(
  name="Indeed Url Creation",
  instructions="""You are a helpful assistant skilled in understanding basic english and human language patterns, you are to interpret the inputs given to you. You will, using the job type, field, example companies, pay, and location to generate a list of urls applicable to the given criteria. You will create a list of several urls, to hit any and all possible job listings which the user may find suitable.

You are not to edit the seek_url_list parsed to you, from the previous agent, only add it to your output without any alteration.

Indeed URLs are in the form of:
https://au.indeed.com/jobs?q=job+title+and+key+words&l=location&from=searchOnHP&vjk=b64c48b125db7894

Examples:
https://au.indeed.com/jobs?q=gym+supplements&l=Brisbane+South+QLD&fromage=14&radius=25&sc=0kf%3Aattr%2875GKK%29%3B&from=searchOnDesktopSerp&vjk=28adbffd6a5a0d74

https://au.indeed.com/jobs?q=analytics&l=St%20Lucia%20QLD&fromage=7&salaryType=%24110%2C000%2B&radius=25&sc=0bf%3Aexrec%28%29%2Ckf%3Aattr%286QC5F%7CEXSNN%7CHFDVW%252COR%29attr%28CF3CP%29attr%28DSQF7%29fcckey%28ad54220b876b7bef%29%3B&from=searchOnDesktopSerp

https://au.indeed.com/jobs?q=supplement&l=brisbane+qld&fromage=last&salaryType=%2490%2C000&from=searchOnDesktopSerp&vjk=e7e7641d6d361c55""",
  model="gpt-4.1",
  output_type=IndeedUrlCreationSchema,
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


class WorkflowInput(BaseModel):
  input_as_text: str


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
  with trace("Job Listing URL developer"):
    state = {

    }
    workflow = workflow_input.model_dump()
    conversation_history: list[TResponseInputItem] = [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": workflow["input_as_text"]
          }
        ]
      }
    ]
    seek_url_creation_result_temp = await Runner.run(
      seek_url_creation,
      input=[
        *conversation_history
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_69a2a3f2a46c8190bd0e23c1c1ccd4710870bf8c03397585"
      })
    )

    conversation_history.extend([item.to_input_item() for item in seek_url_creation_result_temp.new_items])

    seek_url_creation_result = {
      "output_text": seek_url_creation_result_temp.final_output.json(),
      "output_parsed": seek_url_creation_result_temp.final_output.model_dump()
    }
    indeed_url_creation_result_temp = await Runner.run(
      indeed_url_creation,
      input=[
        *conversation_history
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_69a2a3f2a46c8190bd0e23c1c1ccd4710870bf8c03397585"
      })
    )

    conversation_history.extend([item.to_input_item() for item in indeed_url_creation_result_temp.new_items])

    indeed_url_creation_result = {
      "output_text": indeed_url_creation_result_temp.final_output.json(),
      "output_parsed": indeed_url_creation_result_temp.final_output.model_dump()
    }
    end_result = {
    "seek_url_list": list(seek_url_creation_result["output_parsed"]["seek_url_list"].values()),
    "indeed_url_list": list(indeed_url_creation_result["output_parsed"]["indeed_url_list"].values()),
    }
    return end_result
