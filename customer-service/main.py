from agents import function_tool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
from openai import AsyncOpenAI
from types import SimpleNamespace
from guardrails.runtime import load_config_bundle, instantiate_guardrails, run_guardrails
from pydantic import BaseModel

# Tool definitions
@function_tool
def get_retention_offers(customer_id: str, account_type: str, current_plan: str, tenure_months: integer, recent_complaints: bool):
  pass

# Shared client for guardrails and file search
client = AsyncOpenAI()
ctx = SimpleNamespace(guardrail_llm=client)
# Guardrails definitions
jailbreak_guardrail_config = {
  "guardrails": [
    { "name": "Jailbreak", "config": { "model": "gpt-5-nano", "confidence_threshold": 0.7 } }
  ]
}
def guardrails_has_tripwire(results):
    return any((hasattr(r, "tripwire_triggered") and (r.tripwire_triggered is True)) for r in (results or []))

def get_guardrail_safe_text(results, fallback_text):
    for r in (results or []):
        info = (r.info if hasattr(r, "info") else None) or {}
        if isinstance(info, dict) and ("checked_text" in info):
            return info.get("checked_text") or fallback_text
    pii = next(((r.info if hasattr(r, "info") else {}) for r in (results or []) if isinstance((r.info if hasattr(r, "info") else None) or {}, dict) and ("anonymized_text" in ((r.info if hasattr(r, "info") else None) or {}))), None)
    if isinstance(pii, dict) and ("anonymized_text" in pii):
        return pii.get("anonymized_text") or fallback_text
    return fallback_text

async def scrub_conversation_history(history, config):
    try:
        guardrails = (config or {}).get("guardrails") or []
        pii = next((g for g in guardrails if (g or {}).get("name") == "Contains PII"), None)
        if not pii:
            return
        pii_only = {"guardrails": [pii]}
        for msg in (history or []):
            content = (msg or {}).get("content") or []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_text" and isinstance(part.get("text"), str):
                    res = await run_guardrails(ctx, part["text"], "text/plain", instantiate_guardrails(load_config_bundle(pii_only)), suppress_tripwire=True, raise_guardrail_errors=True)
                    part["text"] = get_guardrail_safe_text(res, part["text"])
    except Exception:
        pass

async def scrub_workflow_input(workflow, input_key, config):
    try:
        guardrails = (config or {}).get("guardrails") or []
        pii = next((g for g in guardrails if (g or {}).get("name") == "Contains PII"), None)
        if not pii:
            return
        if not isinstance(workflow, dict):
            return
        value = workflow.get(input_key)
        if not isinstance(value, str):
            return
        pii_only = {"guardrails": [pii]}
        res = await run_guardrails(ctx, value, "text/plain", instantiate_guardrails(load_config_bundle(pii_only)), suppress_tripwire=True, raise_guardrail_errors=True)
        workflow[input_key] = get_guardrail_safe_text(res, value)
    except Exception:
        pass

async def run_and_apply_guardrails(input_text, config, history, workflow):
    results = await run_guardrails(ctx, input_text, "text/plain", instantiate_guardrails(load_config_bundle(config)), suppress_tripwire=True, raise_guardrail_errors=True)
    guardrails = (config or {}).get("guardrails") or []
    mask_pii = next((g for g in guardrails if (g or {}).get("name") == "Contains PII" and ((g or {}).get("config") or {}).get("block") is False), None) is not None
    if mask_pii:
        await scrub_conversation_history(history, config)
        await scrub_workflow_input(workflow, "input_as_text", config)
        await scrub_workflow_input(workflow, "input_text", config)
    has_tripwire = guardrails_has_tripwire(results)
    safe_text = get_guardrail_safe_text(results, input_text)
    fail_output = build_guardrail_fail_output(results or [])
    pass_output = {"safe_text": (get_guardrail_safe_text(results, input_text) or input_text)}
    return {"results": results, "has_tripwire": has_tripwire, "safe_text": safe_text, "fail_output": fail_output, "pass_output": pass_output}

def build_guardrail_fail_output(results):
    def _get(name: str):
        for r in (results or []):
            info = (r.info if hasattr(r, "info") else None) or {}
            gname = (info.get("guardrail_name") if isinstance(info, dict) else None) or (info.get("guardrailName") if isinstance(info, dict) else None)
            if gname == name:
                return r
        return None
    pii, mod, jb, hal, nsfw, url, custom, pid = map(_get, ["Contains PII", "Moderation", "Jailbreak", "Hallucination Detection", "NSFW Text", "URL Filter", "Custom Prompt Check", "Prompt Injection Detection"])
    def _tripwire(r):
        return bool(r.tripwire_triggered)
    def _info(r):
        return r.info
    jb_info, hal_info, nsfw_info, url_info, custom_info, pid_info, mod_info, pii_info = map(_info, [jb, hal, nsfw, url, custom, pid, mod, pii])
    detected_entities = pii_info.get("detected_entities") if isinstance(pii_info, dict) else {}
    pii_counts = []
    if isinstance(detected_entities, dict):
        for k, v in detected_entities.items():
            if isinstance(v, list):
                pii_counts.append(f"{k}:{len(v)}")
    flagged_categories = (mod_info.get("flagged_categories") if isinstance(mod_info, dict) else None) or []
    
    return {
        "pii": { "failed": (len(pii_counts) > 0) or _tripwire(pii), "detected_counts": pii_counts },
        "moderation": { "failed": _tripwire(mod) or (len(flagged_categories) > 0), "flagged_categories": flagged_categories },
        "jailbreak": { "failed": _tripwire(jb) },
        "hallucination": { "failed": _tripwire(hal), "reasoning": (hal_info.get("reasoning") if isinstance(hal_info, dict) else None), "hallucination_type": (hal_info.get("hallucination_type") if isinstance(hal_info, dict) else None), "hallucinated_statements": (hal_info.get("hallucinated_statements") if isinstance(hal_info, dict) else None), "verified_statements": (hal_info.get("verified_statements") if isinstance(hal_info, dict) else None) },
        "nsfw": { "failed": _tripwire(nsfw) },
        "url_filter": { "failed": _tripwire(url) },
        "custom_prompt_check": { "failed": _tripwire(custom) },
        "prompt_injection": { "failed": _tripwire(pid) },
    }
class ClassificationAgentSchema(BaseModel):
  classification: str


classification_agent = Agent(
  name="Classification agent",
  instructions="""Classify the user’s intent into one of the following categories: \"return_item\", \"cancel_subscription\", or \"get_information\". 

1. Any device-related return requests should route to return_item.
2. Any retention or cancellation risk, including any request for discounts should route to cancel_subscription.
3. Any other requests should go to get_information.""",
  model="gpt-4.1-mini",
  output_type=ClassificationAgentSchema,
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


return_agent = Agent(
  name="Return agent",
  instructions="""Offer a replacement device with free shipping.
""",
  model="gpt-4.1-mini",
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


retention_agent = Agent(
  name="Retention Agent",
  instructions="You are a customer retention conversational agent whose goal is to prevent subscription cancellations. Ask for their current plan and reason for dissatisfaction. Use the get_retention_offers to identify return options. For now, just say there is a 20% offer available for 1 year.",
  model="gpt-4.1-mini",
  tools=[
    get_retention_offers
  ],
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    parallel_tool_calls=True,
    max_tokens=2048,
    store=True
  )
)


information_agent = Agent(
  name="Information agent",
  instructions="""You are an information agent for answering informational queries. Your aim is to provide clear, concise responses to user questions. Use the policy below to assemble your answer.

Company Name: HorizonTel Communications Industry: Telecommunications Region: North America
📋 Policy Summary: Mobile Service Plan Adjustments
Policy ID: MOB-PLN-2025-03 Effective Date: March 1, 2025 Applies To: All residential and small business mobile customers
Purpose: To ensure customers have transparent and flexible options when modifying or upgrading their existing mobile service plans.
🔄 Plan Changes & Upgrades
Eligibility: Customers must have an active account in good standing (no outstanding balance > $50).
Upgrade Rules:
Device upgrades are permitted once every 12 months if the customer is on an eligible plan.
Early upgrades incur a $99 early-change fee unless the new plan’s monthly cost is higher by at least $15.
Downgrades: Customers can switch to a lower-tier plan at any time; changes take effect at the next billing cycle.
CS Rep Tip: When customers request plan changes, confirm their next billing cycle and remind them that prorated charges may apply. Always check for active device installment agreements before confirming a downgrade.
💰 Billing & Credits
Billing Cycle: Monthly, aligned with the activation date.
Credit Adjustments:
Overcharges under $10 are automatically credited to the next bill.
For amounts >$10, open a “Billing Adjustment – Tier 2” ticket for supervisor review.
Refund Policy:
Refunds are issued to the original payment method within 7–10 business days.
For prepaid accounts, credits are applied to the balance—no cash refunds.
CS Rep Tip: If a customer reports a billing discrepancy within 30 days, you can issue an immediate one-time goodwill credit (up to $25) without manager approval.
🛜 Network & Outage Handling
Planned Maintenance: Customers receive SMS alerts for outages >1 hour.
Unplanned Outages:
Check the internal “Network Status Dashboard” before escalating.
If multiple customers in a region report the same issue, tag the ticket as “Regional Event – Network Ops.”
Compensation: Customers experiencing service interruption exceeding 24 consecutive hours are eligible for a 1-day service credit upon request.
📞 Retention & Cancellations
Notice Period: 30 days for postpaid accounts; immediate for prepaid.
Retention Offers:
Agents may offer up to 20% off the next 3 billing cycles if the customer cites “cost concerns.”
Retention codes must be logged under “RET-SAVE20.”
Cancellation Fee:
Applies only to term contracts (usually $199 flat rate).
Fee waived for verified relocation to non-serviceable area.
CS Rep Tip: Before processing a cancellation, review alternative retention offers—customers frequently stay when offered a temporary discount or bonus data package.
🧾 Documentation Checklist for CS Reps
Verify customer ID and account number.
Check account standing (billing, contracts, upgrades).
Record all interactions in the CRM ticket.
Confirm next billing cycle date for any changes.
Apply standard note template:
“Customer requested [plan/billing/support] change. Informed of applicable fees, next cycle adjustment, and confirmation reference #[ticket].”
⚠️ Compliance & Privacy
All interactions must comply with CCPA and FCC privacy standards.
Do not record or store personal payment information outside the secure billing system.
Use the “Secure Verification Flow” for identity confirmation before discussing account details.
🧠 Example""",
  model="gpt-4.1-mini",
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


def approval_request(message: str):
  # TODO: Implement
  return True

class WorkflowInput(BaseModel):
  input_as_text: str


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
  with trace("Customer service"):
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
    guardrails_input_text = workflow["input_as_text"]
    guardrails_result = await run_and_apply_guardrails(guardrails_input_text, jailbreak_guardrail_config, conversation_history, workflow)
    guardrails_hastripwire = guardrails_result["has_tripwire"]
    guardrails_anonymizedtext = guardrails_result["safe_text"]
    guardrails_output = (guardrails_hastripwire and guardrails_result["fail_output"]) or guardrails_result["pass_output"]
    if guardrails_hastripwire:
      return guardrails_output
    else:
      classification_agent_result_temp = await Runner.run(
        classification_agent,
        input=[
          *conversation_history
        ],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_69f47a1d9a5c8190a35494943bfb2fc30e7ee21fb09186a6"
        })
      )

      conversation_history.extend([item.to_input_item() for item in classification_agent_result_temp.new_items])

      classification_agent_result = {
        "output_text": classification_agent_result_temp.final_output.json(),
        "output_parsed": classification_agent_result_temp.final_output.model_dump()
      }
      if classification_agent_result["output_parsed"]["classification"] == "return_item":
        return_agent_result_temp = await Runner.run(
          return_agent,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_69f47a1d9a5c8190a35494943bfb2fc30e7ee21fb09186a6"
          })
        )

        conversation_history.extend([item.to_input_item() for item in return_agent_result_temp.new_items])

        return_agent_result = {
          "output_text": return_agent_result_temp.final_output_as(str)
        }
        approval_message = "Does this work for you?"

        if approval_request(approval_message):
            end_result = {
              "message": "Your return is on the way."
            }
            return end_result
        else:
            end_result = {
              "message": "What else can I help you with?"
            }
            return end_result
      elif classification_agent_result["output_parsed"]["classification"] == "cancel_subscription":
        retention_agent_result_temp = await Runner.run(
          retention_agent,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_69f47a1d9a5c8190a35494943bfb2fc30e7ee21fb09186a6"
          })
        )

        conversation_history.extend([item.to_input_item() for item in retention_agent_result_temp.new_items])

        retention_agent_result = {
          "output_text": retention_agent_result_temp.final_output_as(str)
        }
      elif classification_agent_result["output_parsed"]["classification"] == "get_information":
        information_agent_result_temp = await Runner.run(
          information_agent,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_69f47a1d9a5c8190a35494943bfb2fc30e7ee21fb09186a6"
          })
        )

        conversation_history.extend([item.to_input_item() for item in information_agent_result_temp.new_items])

        information_agent_result = {
          "output_text": information_agent_result_temp.final_output_as(str)
        }
      else:
        return classification_agent_result
