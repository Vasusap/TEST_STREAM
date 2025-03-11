import streamlit as st
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine,text
from langchain.sql_database import SQLDatabase
from urllib.parse import quote_plus 
import os
from dotenv import load_dotenv,find_dotenv
from langchain_community.document_loaders import ( PyPDFLoader, TextLoader )
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import HanaDB
from hdbcli import dbapi
_ = load_dotenv(find_dotenv())


st.set_page_config("genAI Scenarios")
st.title("genAI Scenarios")
st.form("Login Form")

## Define RAG Prompt
rag_syst_prompt = ("""
          Answer the user question based on only the given Context,
         if you dont know tell user dont know, dont give your wrong answers and dont give ur own explanation
          {context}              
         """)
rag_prompt = ChatPromptTemplate.from_messages([("system",rag_syst_prompt),("human","{input}"), ])

## Define Chat Prompt
abap_prompt = ChatPromptTemplate.from_messages([
    ("system",
        "You are an ABAP Expert"
        "Write the ABAP Object for only the requested {ABAP_OBJECT} information about given {text} scenario , dont generate the extra information"
        "If you do not know the code, return you dont know . "
        "Return only the ABAP code for the requested object {ABAP_OBJECT} don't generate Note & extra information."
        "don't generate extra Notes & extra information"),
    ("human", "{text}"),
])

## DB Prompt 
hdb_template = """Based on the table schema below, Convert the following question into an SQL query. 
Provide only the SQL code, without explanation or formatting.
write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""

hdb_prompt = ChatPromptTemplate.from_template(hdb_template)

## SQL to NLP Prompt

# This template converts SQL query results into a human-readable response.
sql_2_nlp_template = """Based on the table schema below, question, sql query, and sql response, write a natural language response :
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""

sql_2_nlp_prompt = ChatPromptTemplate.from_template(sql_2_nlp_template)

# to format the output during the LCEL chain for RAG + LLM scenario
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# to execute the SQL Query 
def run_query(query):
    return db.run(query)


# to get the schema of the database
def get_schema(_):
    schema = db.get_table_info()
    return schema
    

def validate_groq_api(api_key):
    """Checks if the provided Groq API Key is valid."""
    try:
        llm = ChatGroq(model="llama3-8b-8192", api_key=api_key)
        llm.invoke("Test")  # Simple test prompt
        return True, "✅ Groq API Key is valid."
    except Exception as e:
        return False, f"❌ Invalid Groq API Key. Error: {str(e)}"

lv_apikey = st.text_input("API Key",placeholder="Enter Valid API Key",type="password")
options = []
if st.button("Validate API Key"):
  if lv_apikey:
    is_valid,message = validate_groq_api(lv_apikey)
    if is_valid:
       st.success(message)
       st.session_state["lv_apikey"] = lv_apikey
       
    else:
       st.error(message)
       
  else:
    st.error("Please Enter your API Key")

if "lv_apikey" in st.session_state:
  st.sidebar.header("Select the Scenarios")
  options = st.sidebar.radio("Select below Options",options=["ABAP","HDB Data","RAG Scenario"])
  if options == "ABAP":
    st.subheader("API Key verified Now Enter your Requirement")
    abap_object_sel =  st.selectbox("ABAP Object",options=["CDS View","AMDP","Program","Class implementation"])
    lv_abap = st.text_area("",height=100,placeholder="ABAP Object Code")
    def sel_abap_object(NONE):
        return abap_object_sel
          
    
    if st.button(f"Fetch"):
     if options == "ABAP":

      llm_model = ChatGroq(model="llama3-70b-8192",api_key= lv_apikey ) 
      chain = ( RunnablePassthrough.assign( ABAP_OBJECT = sel_abap_object) 
              | abap_prompt
              | llm_model  )
      st.write(abap_object_sel)         
      Response = chain.stream({"text":lv_abap})
      Response
  if options == "HDB Data":
      lv_hdb_input = st.text_area("",height=100,placeholder="Capacity Details from HDB")
      if st.button("Fetch HDB"):
        
        username = os.getenv("uname")
        password   = os.getenv("password")
        password = quote_plus(password)
        host   = os.getenv("host")
        port   = os.getenv("port")
        connection_url = f"hana+hdbcli://{username}:{password}@{host}:{port}/?encrypt=true"
        engine = create_engine(connection_url)
        try:
            with engine.connect( ) as connection:
             st.success("Sucessfully Connected to HDB ")
        except Exception as e:
            st.error("Connection failed:{e}")

        llm_model = ChatGroq(model="llama3-70b-8192",api_key= lv_apikey )
        db = SQLDatabase(engine)
        sql_chain = (
          RunnablePassthrough.assign(schema=get_schema)
        | hdb_prompt
        | llm_model.bind(stop="\nSQLResult:")
        | StrOutputParser() )
        # user_question = "what is teh available capacity of WORK_A"
        # response = sql_chain.invoke({"question":user_question})
        final_chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
        schema=get_schema,
        response=lambda vars : run_query(vars['query']) )
        | sql_2_nlp_prompt
        | llm_model
        | StrOutputParser() )
        response = final_chain.invoke({"question":lv_hdb_input})
        response
                 

        
  if options == "RAG Scenario":
        
        
        lv_hdb = st.text_area("",height=100,placeholder="Ask anything based on Uploaded context")
       # file_upload = st.file_uploader("Upload Files",accept_multiple_files=True,type=["txt"])
        file_upload = "data/test_data.txt"
        if st.button("Retrive Data") and file_upload is not None:
         loader = TextLoader(file_upload)
         data = loader.load()
         data_content = "\n".join([data_cont.page_content for data_cont in data])
         txt_split = RecursiveCharacterTextSplitter( chunk_size = 1000, chunk_overlap = 100, separators=["\n\n","\n"," ",""])
         chunks = txt_split.split_text(data_content)
         embeddings = GoogleGenerativeAIEmbeddings(
         model="models/embedding-001",  # Gemini embedding model
         google_api_key=os.getenv("GOOGLE_API_KEY")  ) # Load API key 

         username = os.getenv("uname")
         password   = os.getenv("password")
         password = quote_plus(password)
         host   = os.getenv("host")
         port   = os.getenv("port")

         hana_conn = dbapi.connect(
		  address="0afe4760-7754-4b54-9b40-9aada4b6e049.hana.trial-us10.hanacloud.ondemand.com",
		  port="443",
		  user="DBADMIN",
		  password="Test@123",
		  autocommit=True )
         
         db = HanaDB( embedding=embeddings, connection=hana_conn, table_name="SAP_ATP_DB")
         db.delete(filter={})

         db.add_texts(chunks)
         
         #data = db.similarity_search(lv_hdb,k=2)
         hdb_retriver = db.as_retriever()
         llm_model = ChatGroq(model="llama3-70b-8192",api_key= lv_apikey )
         

         rag_chain =({
            'context':hdb_retriver | format_docs,
            "input":RunnablePassthrough()} | rag_prompt | llm_model | StrOutputParser() )
         response = rag_chain.stream(lv_hdb)
         response
             
         

         
         
           

     
      

