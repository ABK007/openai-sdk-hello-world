import chainlit as cl
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv
import os

load_dotenv() 

gemini_api_key = os.getenv("GEMINI_API_KEY")


# step 1: provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# step 2: model
model = OpenAIChatCompletionsModel(
    openai_client=provider,
    model="gemini-2.0-flash",
)

# Config: Defined at Run Level
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
    )

# step 3: agent
agent = Agent(
    instructions="You are a helpful assistant that can asnwer questions",
    name="Panaversity Support Agent"
)

# step 4: run agent
result = Runner.run_sync(
    input="What is the capital of France?",
    run_config=run_config,
    starting_agent=agent
)

print(result)


