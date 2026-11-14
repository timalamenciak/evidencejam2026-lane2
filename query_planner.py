
import os
import json
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["BRINE_URL"],
    api_key=os.environ["BRINE_API_KEY"],
    timeout=120.0,
    max_retries=0
)


def plan_question(question):

    prompt = f"""
You are a query planner for a scientific causal graph.

Your job is NOT to answer the question.

Instead, identify the concepts we should search for
in a graph of ecological causal relationships.

QUESTION:
{question}

Return ONLY a valid JSON object with these keys:

{{
    "interventions": ["..."],
    "outcomes": ["..."],
    "subquestions": ["..."],
    "synonyms": {{
        "concept": ["alternative term 1", "alternative term 2"]
    }}
}}

RULES:
- Use short scientific search terms.
- Include important interventions and outcomes.
- Generate useful synonyms, but avoid unrelated concepts.
- Break complex questions into smaller subquestions.
- Do not invent evidence or scientific findings.
- Do not include Markdown formatting.
"""

    print("Qwen is interpreting the question...", flush=True)

    response = client.chat.completions.create(
        model=os.environ["BRINE_MODEL"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You convert ecological questions into "
                    "structured graph search plans. "
                    "Return only JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1500
    )

    choice = response.choices[0]
    content = choice.message.content

    if not content:
        print("Finish reason:", choice.finish_reason)
        raise ValueError(
            "Qwen returned no JSON content. "
            "The model may have exhausted its token limit."
        )

    # Remove Markdown fences if the model adds them
    content = content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0].strip()

    return json.loads(content)


if __name__ == "__main__":

    question = (
        "Which restoration actions, such as peatland rewetting, "
        "moss layer transfer or salt marsh realignment, "
        "turn degraded sites back into carbon sinks, "
        "and what happens to methane emissions?"
    )

    plan = plan_question(question)

    print("\nGENERATED SEARCH PLAN:\n")
    print(json.dumps(plan, indent=2))
