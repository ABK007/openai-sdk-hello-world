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
# result = Runner.run_sync(
#     input="What is the capital of France?",
#     run_config=run_config,
#     starting_agent=agent
# )  -w

# print(result)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello, I'm Panaversity Support Agent. How can I help you today?").send()
    

@cl.on_message
async def handle_message(message: cl.message):
    history = cl.user_session.get("history")
    
    history.append(
        {
            "role": "user",
            "content": message.content
        }
    )
    result = await Runner.run(
        starting_agent=agent,
        input=history,
        run_config=run_config,
    )
    
    history.append(
        {
            "role": "assistant",
            "content": result.final_output
        }
    )
    
    cl.user_session.set("history", history)
    
    await cl.Message(content=result.final_output).send()

