import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from openai import OpenAI
from neo4j import GraphDatabase


load_dotenv()

import warnings
warnings.filterwarnings("ignore")

AURA_INSTANCENAME=os.environ["AURA_INSTANCENAME"]
NEO4J_URI= os.environ["NEO4J_URI"]
NEO4J_USERNAME= os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD= os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE= os.environ["NEO4J_DATABASE"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print(client)

# ---- Step 1: Generate Cypher from Prompt ----
def generate_cypher_from_prompt(prompt):
    system_prompt = """
You are a Cypher expert for a Neo4j database containing SAP Accounts Receivable data.

The graph contains these nodes and relationships:
- (Company {company_code, company_name})
- (Customer {customer_id, customer_name})
- (Invoice {invoice_id, invoiced_amount, invoiced_date, invoiced_status})
- (Customer_Payment {cust_payment_id, cust_payment_amt, cust_payment_date})
- (FI_Document {fidoc_number, fidoc_fisc_year, fidoc_type})
- (Revenue_GL {revgl_acct, revgl_desc})
- (Customer_Payment {cust_payment_id, cust_payment_amt, cust_payment_date})
- (Bank_GL_Acct {bank_gl_acct, bank_name})
- (Dunning {dunning_id, dunning_level, dunning_date})
- (Sales_Order {sales_ord_id, sales_ord_date, sales_ord_amt, sales_ord_status})
- (Delivery {delivery_id, delivery_date, delivery_status})

- Relationships include:
  - (Customer)-[:BELONGS_TO]->(Company)
  - (Invoice)-[:ISSUED_TO]->(Customer)
  - (Invoice)-[:POSTED_AS]->(FI_Document)
  - (Customer_Payment)<-[:MADE_PAYMENT]-(Customer)
  - (FI_Document)-[:USES_ACCOUNT]->(Revenue_GL)
  - (Customer_Payment)-[:CLEARS]->(Invoice)  
  - (Sales_Order)-[:CREATED_FOR]->(Customer)
  - (Delivery)-[:FULFILLS]->(Sales_Order)
  - (Invoice)-[:BILLED_FOR]->(Delivery) 

Generate a Cypher query to answer the user's prompt. 
Return only the Cypher query (no explanation, no code formatting).
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"

# ---- Step 2: Run Cypher Query ----
def run_cypher_query(query):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query)
            records = [dict(record) for record in result]
            return records
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        driver.close()

# ---- Step 3: Streamlit UI ----
st.set_page_config(page_title="SAP AR Graph Agent", layout="centered")
st.title("🔍 SAP AR Query Agent with Neo4j + GPT")
st.markdown("""
Ask natural questions like:
- `Show all unpaid invoices by customer for 2023`
- `Total invoiced amount for company code 1000`
- `Invoices linked to FI document type DR`
- `Payments received in April 2024 by company 2000`
""")

user_input = st.text_input("Ask your question:", placeholder="e.g. List invoices posted as FI doc type DR")

if st.button("Run Query") and user_input.strip():
    with st.spinner("🔍 Generating Cypher query using GPT..."):
        cypher_query = generate_cypher_from_prompt(user_input)

        if cypher_query.startswith("ERROR"):
            st.error(cypher_query)
        else:
            st.code(cypher_query, language="cypher")

            with st.spinner("📡 Querying Neo4j..."):
                results = run_cypher_query(cypher_query)

                if results and "error" in results[0]:
                    st.error(f"❌ Cypher Error: {results[0]['error']}")                                       
                elif results:
                    st.success("✅ Query successful. Here are the results:")
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                    # ---- Step 4: Auto Chart if numeric columns exist ----
                    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
                    if len(numeric_cols) > 0 and len(df.columns) >= 2:
                        x_col = df.columns[0]
                        y_col = numeric_cols[0]
                        st.markdown(f"📈 **Chart: {y_col} by {x_col}**")
                        st.bar_chart(data=df, x=x_col, y=y_col)

                else:
                    st.info("✅ No results found.")