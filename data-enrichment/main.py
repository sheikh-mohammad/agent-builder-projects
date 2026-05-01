from pydantic import BaseModel
from agents import Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
from openai.types.shared.reasoning import Reasoning

class WebResearchAgentSchema__CompaniesItem(BaseModel):
  company_name: str
  industry: str
  headquarters_location: str
  company_size: str
  website: str
  description: str
  founded_year: float


class WebResearchAgentSchema(BaseModel):
  companies: list[WebResearchAgentSchema__CompaniesItem]


class SummarizeAndDisplaySchema(BaseModel):
  company_name: str
  industry: str
  headquarters_location: str
  company_size: str
  website: str
  description: str
  founded_year: float


web_research_agent = Agent(
  name="Web research agent",
  instructions="You are a helpful assistant. Use web search to find information about the following company I can use in marketing asset based on the underlying topic.",
  model="gpt-5-mini",
  output_type=WebResearchAgentSchema,
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="low",
      summary="auto"
    )
  )
)


summarize_and_display = Agent(
  name="Summarize and display",
  instructions="""Put the research together in a nice display using the output format described.
""",
  model="gpt-5",
  output_type=SummarizeAndDisplaySchema,
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="minimal",
      summary="auto"
    )
  )
)


class WorkflowInput(BaseModel):
  input_as_text: str


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
  with trace("Data enrichment"):
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
    web_research_agent_result_temp = await Runner.run(
      web_research_agent,
      input=[
        *conversation_history
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_69f477ca3e0c8190b116f5a78cf2d7820c98dae53e6426c3"
      })
    )

    conversation_history.extend([item.to_input_item() for item in web_research_agent_result_temp.new_items])

    web_research_agent_result = {
      "output_text": web_research_agent_result_temp.final_output.json(),
      "output_parsed": web_research_agent_result_temp.final_output.model_dump()
    }
    summarize_and_display_result_temp = await Runner.run(
      summarize_and_display,
      input=[
        *conversation_history
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_69f477ca3e0c8190b116f5a78cf2d7820c98dae53e6426c3"
      })
    )
    summarize_and_display_result = {
      "output_text": summarize_and_display_result_temp.final_output.json(),
      "output_parsed": summarize_and_display_result_temp.final_output.model_dump()
    }
