from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# FIX: Use .invoke() instead of calling it directly
response = llm.invoke("What is the capital of India?")

# FIX: Print response.content to see just the text answer
print(response.content)


