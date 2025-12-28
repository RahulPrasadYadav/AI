from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.0-flash")

import streamlit as st
temp=PromptTemplate(

   template=""""
   your are the expert in the field of {field}
and temm me about {topic} in {language} language.
 
minumin {length} words.
   
   """,
  
  input_variables=["field","topic","language","length"]

)



st.header("Dynamic Practice Prompt")


field_input = st.selectbox("Enter the Field of Expertise",["Artificial Intelligence", "Quantum Computing", "Renewable Energy", "Blockchain Technology","Machine Learning","math"] )

topic_input = st.selectbox("Enter the Topic to Explain",["Neural Networks", "Quantum Entanglement", "Solar Power Technologies", "Cryptocurrency Mechanisms","Supervised Learning","calculus"] )
language_input = st.selectbox("Enter the Language for Explanation",["English", "Spanish", "French", "German","Chinese","hindi"] )
length_input = st.selectbox("Enter the Minimum Length (in words)", ["50", "100", "200", "500","1000"]   )  

if st.button("Generate Explanation"):
    chain = temp | model
    result = chain.invoke({
        'field': field_input,
        'topic': topic_input,
        'language': language_input,
        'length': length_input
    })

    st.write(result.content)