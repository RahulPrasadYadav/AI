from langchain_openai import OpenAIEmbeddings


from dotenv import load_dotenv
load_dotenv()   



embedding=OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

docoment=[

      "india is best  countery in the world .. ",
      "virat kohali is .goat batsman",
      "delhi captital of india"
]


result=embedding.embed_documents(docoment)


print(result)
print(len(result))