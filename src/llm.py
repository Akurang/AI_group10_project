import re

import os

# Your own prompt from driver.ipynb (EPXLANAION_PROMPT), lightly adapted so it
# receives the already-computed probability/explanation instead of asking the
# LLM to interpret a raw dataframe + a bare 0/1 label itself.

# SYSTEM_PROMPT = "You are a business advisor for an ecommerce chain with an expertise in senior customer retention manager."

SYSTEM_PROMPT = ("You are a business advisor for an ecommerce chain specializing in customer retention. "
    "CRITICAL: Do NOT use <think> tags. Do NOT show your internal reasoning or thinking process. "
    "Provide your final answer immediately, structured strictly into the 3 requested points.")

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
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=700,
        )
        raw_content = response.choices[0].message.content
        clean_content = re.sub(r"<think>.*?(?:</think>|$)", "", raw_content, flags=re.DOTALL).strip()

        # If the model spent all its tokens thinking and returned nothing, give a clean UI message
        if not clean_content:
            return (
                "⚠️ **The AI spent too long analyzing the data.**\n\n"
                "Please click the **Generate AI Strategy** button again to retry. "
                "The system has adjusted the parameters to force a direct answer."
            )

        return clean_content
    

    
    except Exception as e:
        # Change 'error' to 'str(e)' so Python knows what variable to read
        print(f"DEBUG: Groq connection failed: {str(e)}")
        driver_names = [d["feature"] if isinstance(d, dict) else str(d) for d in drivers]
        return (
            f" **Note: AI recommendation engine is currently offline.**\n\n"
            f"**Automated Action Plan for {risk_label}:**\n"
            
            f"2. System logs for error: `{str(e)}`"
        )

        

