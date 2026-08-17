

import os

# Your own prompt from driver.ipynb (EPXLANAION_PROMPT), lightly adapted so it
# receives the already-computed probability/explanation instead of asking the
# LLM to interpret a raw dataframe + a bare 0/1 label itself.
SYSTEM_PROMPT = "You are a business advisor for an ecommerce chain."

USER_PROMPT_TEMPLATE = """Take the following customer's profile and churn result (1 = leave, 0 = stay) and output:
1. Reasons why they might leave if value is 1, or reasons they might stay if value is 0
2. Possible steps to make the customer stay
3. Business advice to ensure the customer's churn value remains at 0, meaning they do not leave

Ensure that it is clear and straightforward.

Customer profile:
{profile_lines}

Churn result: {churn_value} ({risk_label})
Churn probability: {probability:.0%}

Top contributing factors (from local explanation):
{driver_lines}
"""


def _build_user_prompt(customer_profile: dict, probability: float, risk_label: str, drivers: list) -> str:
    churn_value = 1 if risk_label == "HIGH" else 0
    driver_lines = "\n".join(
        f"- {d['friendly_name']} (contribution: {d['contribution']:.0%}, actionable={d['actionable']})"
        for d in drivers
    ) or "- No strong individual drivers identified"

    profile_lines = "\n".join(f"- {k}: {v}" for k, v in customer_profile.items())

    return USER_PROMPT_TEMPLATE.format(
        profile_lines=profile_lines,
        churn_value=churn_value,
        risk_label=risk_label,
        probability=probability,
        driver_lines=driver_lines,
    )


def generate_retention_strategy(customer_profile: dict, probability: float, risk_label: str, drivers: list) -> str:
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return " Error: GROQ_API_KEY is missing from system environment variables."
    
    user_prompt = _build_user_prompt(customer_profile, probability, risk_label, drivers)


    try:
        from openai import OpenAI  # Groq exposes an OpenAI-compatible client

        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1,
            max_tokens=500,
        )
        return response.choices[0].message.content

    
    except Exception as e:
        # Change 'error' to 'str(e)' so Python knows what variable to read
        print(f"DEBUG: Groq connection failed: {str(e)}")
        driver_names = [d["feature"] if isinstance(d, dict) else str(d) for d in drivers]
        return (
            f"⚠️ **Note: AI recommendation engine is currently offline.**\n\n"
            f"**Automated Action Plan for {risk_label}:**\n"
            
            f"2. System logs for error: `{str(e)}`"
        )
        print(f"DEBUG: Groq connection failed: {str(e)}")
        



    driver_names = [d["friendly_name"] for d in drivers if d.get("actionable")] or [
        d["friendly_name"] for d in drivers
    ]
    header = "**[Demo mode -- no GROQ_API_KEY set, showing a template response]**\n\n"
    if error:
        header = f"**[LLM call failed: {error} -- showing a template response]**\n\n"

    if risk_label == "HIGH":
        body = f"""**Why this customer may churn**
This customer's risk factors center on {', '.join(driver_names[:2]) if driver_names else 'reduced recent engagement'}, suggesting declining satisfaction with the service.

**Recommended actions**
1. Offer a personalized discount or cashback bonus on their next order
2. Proactively follow up on any unresolved complaint
3. Recommend products related to their previous purchase category
4. Enroll them in a targeted re-engagement email/SMS campaign

**Business advice**
Prioritize this customer for immediate outreach -- their predicted churn probability exceeds the intervention threshold."""
    else:
        body = """**Why this customer is likely to stay**
This customer's behavior looks broadly consistent with the retained-customer population -- steady ordering activity and no major red flags.

**Recommended actions**
1. Continue standard loyalty engagement (points, seasonal offers)
2. Monitor for any drop in order frequency
3. Consider light-touch upsell based on purchase history

**Business advice**
No urgent action needed; keep this customer in the standard engagement track."""

    return header + body
