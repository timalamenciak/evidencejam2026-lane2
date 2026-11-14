
import os
import json
from openai import OpenAI
from retrieval import search_question, nodes


# Connect to the Qwen API
client = OpenAI(
    base_url=os.environ["BRINE_URL"],
    api_key=os.environ["BRINE_API_KEY"],
    timeout=120.0,
    max_retries=0
)


def reason_about_question(question):

     # Step 1: Retrieve candidate evidence
    print("Searching the causal graph...", flush=True)

    edges = search_question(question, top_k=5)

    print(f"Found {len(edges)} candidate edges.", flush=True)

    # NEW: Display the exact evidence sent to Qwen
    print("\nRETRIEVED EDGE IDS:")

    for edge in edges:
        print(edge["id"])

    evidence = []

    for edge in edges:
        subject = nodes[edge["subject"]]
        outcome = nodes[edge["object"]]

        evidence.append({
            "edge_id": edge["id"],
            "subject": subject.get("name"),
            "subject_qualifier": subject.get(
                "state_or_change_qualifier"
            ),
            "predicate": edge.get("predicate"),
            "outcome": outcome.get("name"),
            "outcome_qualifier": outcome.get(
                "state_or_change_qualifier"
            ),
            "causal_language": edge.get("causal_language"),
            "negated": edge.get("negated"),
            "certainty": edge.get("certainty_grade"),
            "source": edge.get("original_sentence")
        })

    # Step 2: Prepare the evidence for Qwen
    evidence_text = json.dumps(evidence, indent=2)

    prompt = f"""
You are a scientific causal reasoning assistant.

QUESTION:
{question}

RETRIEVED EVIDENCE:
{evidence_text}

TASK:
1. Identify which edges directly help answer the question.
2. Explain what each relevant edge establishes.
3. Distinguish causation from association.
4. Identify conflicting or uncertain evidence.
5. Do not invent evidence or edge IDs.
6. If the evidence is insufficient, explicitly say so.

Return a concise answer with supporting edge IDs.
"""

    # Step 3: Send the request to Qwen
    print("Sending evidence to Qwen...", flush=True)

    response = client.chat.completions.create(
        model=os.environ["BRINE_MODEL"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evidence-grounded ecological "
                    "reasoning assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1200
    )

    print("Qwen response received!", flush=True)

    # Step 4: Inspect the response
    choice = response.choices[0]
    message = choice.message

    print("\n--- QWEN DEBUG INFORMATION ---")
    print("Finish reason:", choice.finish_reason)
    print("Content:", repr(message.content))

    reasoning_content = getattr(
        message,
        "reasoning_content",
        None
    )

    print(
        "Reasoning content:",
        repr(reasoning_content)[:1000]
    )

    print("Tool calls:", repr(message.tool_calls))

    if response.usage:
        print("Token usage:", response.usage)

    print("------------------------------\n")

    # Step 5: Return the final answer
    return message.content


if __name__ == "__main__":

    question = (
        "Which restoration actions, such as peatland rewetting, "
        "moss layer transfer or salt marsh realignment, "
        "turn degraded sites back into carbon sinks, "
        "and what happens to methane emissions?"
    )

    answer = reason_about_question(question)

    print("\nQWEN ANSWER:\n")

    if answer:
        print(answer)
    else:
        print(
            "No final text answer was returned. "
            "Check the debug information above."
        )
