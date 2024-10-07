from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
import streamlit as st

examplesList = [
    {
        "input": "List all projects in started status",
        "query": "select count(*) from projects where status = 'started';"
    },
    {
        "input": "List all the program name",
        "query": "SELECT DISTINCT(programname) as Program_Name from solutions;"
    },
    {
        "input": "Can you explain more about PRG228 program",
        "query": "SELECT programdescription from solutions where programname = 'PRG228';"
    }
]

# vectorstore = Chroma()
# vectorstore.delete_collection()
# print(f"Collection initialized: {vectorstore.collection}")

@st.cache_resource
def get_example_selector():
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examplesList,
        OpenAIEmbeddings(),
        Chroma,
        k=2,
        input_keys=["input"],
    )
    return example_selector