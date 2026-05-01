from agents import WebSearchTool, CodeInterpreterTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
from pydantic import BaseModel
from openai.types.shared.reasoning import Reasoning

# Tool definitions
web_search_preview = WebSearchTool(
  search_context_size="medium",
  user_location={
    "type": "approximate"
  }
)
code_interpreter = CodeInterpreterTool(tool_config={
  "type": "code_interpreter",
  "container": {
    "type": "auto",
    "file_ids": [

    ]
  }
})
class ClassifySchema(BaseModel):
  operating_procedure: str


query_rewrite = Agent(
  name="Query rewrite",
  instructions="Rewrite the user's question to be more specific and relevant to the knowledge base.",
  model="gpt-5",
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="low",
      summary="auto"
    )
  )
)


classify = Agent(
  name="Classify",
  instructions="Determine whether the question should use the Q&A or fact-finding process.",
  model="gpt-5",
  output_type=ClassifySchema,
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="low",
      summary="auto"
    )
  )
)


internal_q_a = Agent(
  name="Internal Q&A",
  instructions="Answer the user's question using the knowledge tools you have on hand (file or web search). Be concise and answer succinctly, using bullet points and summarizing the answer up front",
  model="gpt-5",
  tools=[
    web_search_preview
  ],
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="low",
      summary="auto"
    )
  )
)


external_fact_finding = Agent(
  name="External fact finding",
  instructions="""Explore external information using the tools you have (web search, file search, code interpreter). 
Analyze any relevant data, checking your work.

Make sure to output a concise answer followed by summarized bullet point of supporting evidence""",
  model="gpt-5",
  tools=[
    web_search_preview,
    code_interpreter
  ],
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="low",
      summary="auto"
    )
  )
)


agent = Agent(
  name="Agent",
  instructions="Ask the user to provide more detail so you can help them by either answering their question or running data analysis relevant to their query",
  model="gpt-4.1-nano",
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
  with trace("Internal knowledge assistant"):
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
    query_rewrite_result_temp = await Runner.run(
      query_rewrite,
      input=[
        *conversation_history,
        {
          "role": "user",
          "content": [
            {
              "type": "input_text",
              "text": f"Original question: {workflow["input_as_text"]}"
            }
          ]
        }
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_69f48e1d251c81909b3a6a3eb5db979d0e5980085aee6f46"
      })
    )

    conversation_history.extend([item.to_input_item() for item in query_rewrite_result_temp.new_items])

    query_rewrite_result = {
      "output_text": query_rewrite_result_temp.final_output_as(str)
    }
    classify_result_temp = await Runner.run(
      classify,
      input=[
        *conversation_history,
        {
          "role": "user",
          "content": [
            {
              "type": "input_text",
              "text": f"Question: {input["output_text"]}"
            }
          ]
        }
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_69f48e1d251c81909b3a6a3eb5db979d0e5980085aee6f46"
      })
    )

    conversation_history.extend([item.to_input_item() for item in classify_result_temp.new_items])

    classify_result = {
      "output_text": classify_result_temp.final_output.json(),
      "output_parsed": classify_result_temp.final_output.model_dump()
    }
    if classify_result["output_parsed"]["operating_procedure"] == "q-and-a":
      internal_q_a_result_temp = await Runner.run(
        internal_q_a,
        input=[
          *conversation_history
        ],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_69f48e1d251c81909b3a6a3eb5db979d0e5980085aee6f46"
        })
      )

      conversation_history.extend([item.to_input_item() for item in internal_q_a_result_temp.new_items])

      internal_q_a_result = {
        "output_text": internal_q_a_result_temp.final_output_as(str)
      }
    elif classify_result["output_parsed"]["operating_procedure"] == "fact-finding":
      external_fact_finding_result_temp = await Runner.run(
        external_fact_finding,
        input=[
          *conversation_history
        ],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_69f48e1d251c81909b3a6a3eb5db979d0e5980085aee6f46"
        })
      )

      conversation_history.extend([item.to_input_item() for item in external_fact_finding_result_temp.new_items])

      external_fact_finding_result = {
        "output_text": external_fact_finding_result_temp.final_output_as(str)
      }
    else:
      agent_result_temp = await Runner.run(
        agent,
        input=[
          *conversation_history
        ],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_69f48e1d251c81909b3a6a3eb5db979d0e5980085aee6f46"
        })
      )

      conversation_history.extend([item.to_input_item() for item in agent_result_temp.new_items])

      agent_result = {
        "output_text": agent_result_temp.final_output_as(str)
      }
