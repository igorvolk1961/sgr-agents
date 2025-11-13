import json

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8010/v1", api_key="dummy")

# Step 1: Initial research request
print("Starting research...")
response = client.chat.completions.create(
    model="sgr_agent_yandex",
#    model="sgr_agent",
    messages=[{"role": "user", "content": "Research BMW X6 2025 prices in Russia"}],
#    messages=[{"role": "user", "content": "Research AI market trends"}],
    stream=True,
    temperature=0,
)

agent_id = None
clarification_questions = []

# Process streaming response
for chunk in response:
    # Extract agent ID from model field
    if chunk.model and chunk.model.startswith("sgr_agent_"):
        agent_id = chunk.model

    # Check for clarification requests
    if chunk.choices and chunk.choices[0].delta.tool_calls:
        print(f"\nAgent ID: {agent_id}")
        for tool_call in chunk.choices[0].delta.tool_calls:
            if tool_call.function and tool_call.function.name == "clarification":
                args = json.loads(tool_call.function.arguments)
                clarification_questions = args.get("questions", [])

    # Print content
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Step 2: Handle clarification if needed
if clarification_questions and agent_id:
    print("\n\nClarification needed:")
    for i, question in enumerate(clarification_questions, 1):
        print(f"{i}. {question}")

    # Provide clarification
#    clarification = "Focus on LLM market trends for 2024-2025, global perspective"
    clarification = "Make your own decision"
    print(f"\nProviding clarification: {clarification}")

    # Continue with agent ID
    response = client.chat.completions.create(
        model=agent_id,  # Use agent ID as model
        messages=[{"role": "user", "content": clarification}],
        stream=True,
        temperature=0,
    )

    # Print final response
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

print("\n\nResearch completed!")
