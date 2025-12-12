from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
load_dotenv()


model=ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)


# temp 0  hai to same ouput aayega .. always .. 


result=model.invoke("write 5 line for criketet")


print(result.content)