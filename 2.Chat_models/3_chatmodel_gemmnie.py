from langchain_google_genai import ChatGoogleGenerativeAI


from dotenv import load_dotenv
load_dotenv()


model=ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=1.5, max_completion_tokens=10)


result=model.invoke("write 5 line for criketet")


print(result.content)