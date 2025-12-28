from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

model = ChatOpenAI()

st.header("Research Tool")

user_input = st.text_area("Enter your research topic here")

if st.button("submit"):
    result = model.invoke(f"write detailed research about {user_input} in 20 words")
    st.write(result.content)