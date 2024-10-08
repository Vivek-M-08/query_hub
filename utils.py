import os
from dotenv import load_dotenv
import urllib.parse
import pandas as pd
import streamlit as st
from operator import itemgetter
from typing import List
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from prompts import answer_prompt

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Function to get database credentials from the environment variables
def get_db_credentials(selected_option):
    db_user = os.getenv(f"{selected_option}_DB_USER")
    db_password = os.getenv(f"{selected_option}_DB_PASSWORD")
    db_host = os.getenv(f"{selected_option}_DB_HOST")
    db_port = os.getenv(f"{selected_option}_DB_PORT")
    db_name = os.getenv(f"{selected_option}_DB_NAME")

    # Check if any required credentials are missing
    if not all([db_user, db_password, db_host, db_port, db_name]):
        raise ValueError(f"Configuration for {selected_option} not found in the environment variables.")

    # Ensure db_password is a quoted string
    db_password = urllib.parse.quote(db_password)

    return {
        "db_user": db_user,
        "db_password": db_password,
        "db_host": db_host,
        "db_port": db_port,
        "db_name": db_name
    }

@st.cache_resource
def get_chain(selected_option):
    print("Creating chain")

    credentials = get_db_credentials(selected_option)
    db_user = credentials['db_user']
    db_password = credentials['db_password']
    db_host = credentials['db_host']
    db_port = credentials['db_port']
    db_name = credentials['db_name']

    # Get table descriptions and table extraction chain
    table_details = get_table_details(selected_option)

    print(db_host + " " + db_name + " " + str(db_port) + " " + db_user + " " + db_password)



    if selected_option == 'KATHA':
        # MySQL Database connection
        db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
    else:
        # PostgreSQL Database connection
        db = SQLDatabase.from_uri(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

    # Additional code to use `db` as needed, e.g., fetching tables or executing queries
    # return db



    # SQL Database connection
    # db = SQLDatabase.from_uri(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
    # db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

    print(db_host + " " + db_name + " " + db_port + " "  + db_user + " " +db_password )





    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Extraction chain to identify relevant tables
    table_chain = create_table_extraction_chain(llm, table_details)

    # Generate and execute SQL queries
    generate_query = create_sql_query_chain(llm, db)
    print("************* generate_query *************")
    print(generate_query)
    execute_query = QuerySQLDataBaseTool(db=db)
    print("************* execute_query *************")
    print(execute_query)

    rephrase_answer = answer_prompt | llm | StrOutputParser()

    chain = (
        RunnablePassthrough.assign(tables=table_chain)
        .assign(query=generate_query)
        .assign(result=itemgetter("query") | execute_query)
        | rephrase_answer
    )

    return chain


# Function to extract table details from a CSV
@st.cache_data
def get_table_details(selected_option):
    # Mapping of selected options to CSV file names
    csv_mapping = {
        "MENTORING": "table_desc_mentoring.csv",
        "SCP": "table_desc_scp.csv",
        "PROJECTS": "table_desc_projects.csv",
        "KATHA": "table_desc_katha.csv"
    }

    # Read the corresponding CSV file
    table_description = pd.read_csv(csv_mapping[selected_option])
    table_details = ""
    for index, row in table_description.iterrows():
        table_details += f"Table Name: {row['Table']}\nTable Description: {row['Description']}\n\n"
    
    print("===================================================================")
    print("table_description = " + table_description)
    print("table_details = " + table_details)


    return table_details

# Define a Pydantic model for the table extraction
class Table(BaseModel):
    name: str = Field(description="Name of table in SQL database.")

# Create a table extraction chain using the LLM
def create_table_extraction_chain(llm, table_details):
    table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
    The tables are:
    
    {table_details}
    
    Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""
    
    # return create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt)
    print("****************************")
    print(itemgetter("question"))
    return {"input": itemgetter("question")} | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt)

def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def invoke_chain(question, messages, selected_option):
    chain = get_chain(selected_option)
    history = create_history(messages)
    
    response = chain.invoke({"question": question, "top_k": 3, "messages": history.messages})

    # Add messages to history
    history.add_user_message(question)
    history.add_ai_message(response)

    return response
