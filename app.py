import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq

# 1. Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up Streamlit page configuration
st.set_page_config(page_title="AI Ticket Analytics System", layout="wide")

# 2. Data Ingestion Layer
@st.cache_data
def load_data():
    """Reads the CSV support ticket dataset and parses dates safely."""
    if not os.path.exists("support_tickets.csv"):
        st.error("Error: 'support_tickets.csv' not found in the directory!")
        return pd.DataFrame()
    
    df = pd.read_csv("support_tickets.csv")
    df['created_at'] = pd.to_datetime(df['created_at'])
    return df

df = load_data()

# 3. Anomaly Detection Layer
def detect_anomalies(data):
    """Flags business and operational anomalies based on data constraints."""
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # Anomaly Type A: Unresolved High/Critical priority tickets older than 24 hours
    # Using the most recent date in the dataset as our simulation checkpoint
    latest_snapshot_date = data['created_at'].max()
    
    unresolved_condition = (
        data['status'].isin(['Open', 'Escalated']) & 
        data['priority'].isin(['High', 'Critical'])
    )
    # Check if hours elapsed from ticket creation to snapshot is > 24
    hours_elapsed = (latest_snapshot_date - data['created_at']).dt.total_seconds() / 3600.0
    backlog_anomalies = data[unresolved_condition & (hours_elapsed > 24)].copy()
    
    # Anomaly Type B: Tickets with abnormally long resolution times (Top 5% outlier threshold)
    resolved_tickets = data[data['resolution_time_hrs'].notnull()]
    if not resolved_tickets.empty:
        threshold_hrs = resolved_tickets['resolution_time_hrs'].quantile(0.95)
        duration_anomalies = data[data['resolution_time_hrs'] > threshold_hrs].copy()
    else:
        duration_anomalies = pd.DataFrame()
        
    return backlog_anomalies, duration_anomalies

# 4. User Interface Shell (Streamlit View Layout)
st.title(" End-to-End AI Support Ticket System")
st.subheader("DOTMappers IT Pvt. Ltd. — Technical Assessment")
st.markdown("---")

# Setup sidebar tabs/navigation
menu = st.sidebar.radio("Navigation", ["Data Dashboard", "Run Anomaly Detection", "AI Natural Language Query"])

if menu == "Data Dashboard":
    st.header("Support Tickets Dataset Summary")
    if not df.empty:
        # Display high-level metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tickets", len(df))
        col2.metric("Open Tickets", len(df[df['status'] == 'Open']))
        col3.metric("Escalated Tickets", len(df[df['status'] == 'Escalated']))
        col4.metric("Resolved Tickets", len(df[df['status'] == 'Resolved']))
        
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No data available.")

elif menu == "Run Anomaly Detection":
    st.header("System Operational Anomalies")
    backlog, durations = detect_anomalies(df)
    
    st.subheader("1. Unresolved High/Critical Priority Tickets (> 24 Hours Old)")
    st.markdown(f"**Found {len(backlog)} flagged issues** where urgent user requests remain stagnant.")
    st.dataframe(backlog, use_container_width=True)
    
    st.subheader("2. Resolution Time Outliers (Top 5% Longest Durations)")
    st.markdown(f"Tickets that took an exceptional amount of operational time to clear.")
    st.dataframe(durations, use_container_width=True)

elif menu == "AI Natural Language Query":
    st.header("AI-Powered Data Assistant")
    st.write("Ask any operational question about the ticket dataset, and the AI will analyze it for you.")

    # Check if Groq API key is configured properly
    if not GROQ_API_KEY:
        st.error("Groq API Key not found! Please check your hidden `.env` file configuration.")
    elif df.empty:
        st.warning("The dataset is empty. Please check your data ingestion pipeline.")
    else:
        # User input text box
        user_query = st.text_input(
            "Enter your question:", 
            placeholder="e.g., How many critical tickets are unresolved?"
        )
        
        if user_query:
            with st.spinner("AI is analyzing the data..."):
                try:
                    # Initialize the Groq Cloud Client
                    client = Groq(api_key=GROQ_API_KEY)
                    
                    # Provide system metadata context to guide the LLM's logic
                    dataframe_schema_context = """
                    You are a strict data scientist assistant. Your job is to translate natural language questions into valid Python Pandas code execution strings operating on a DataFrame named 'df'.
                    
                    The DataFrame 'df' contains 500 rows with the following column structure:
                    - ticket_id (str): Unique ticket identifier
                    - created_at (datetime): Ticket creation timestamp
                    - category (str): 'Billing', 'Technical', or 'General'
                    - priority (str): 'Low', 'Medium', 'High', or 'Critical'
                    - status (str): 'Open', 'Resolved', or 'Escalated'
                    - response_time_hrs (float): Hours to first response
                    - resolution_time_hrs (float): Hours to resolution (null if unresolved)
                    - agent_id (str): Support agent identifier
                    - customer_rating (float): Satisfaction rating 1-5 (null if unresolved)
                    - issue_summary (str): Short description text
                    
                    CRITICAL INSTRUCTIONS:
                    1. Return ONLY the executable python expression. Do NOT write blocks of explanation, markdown code ticks, or text. 
                    2. Example Good Outputs:
                       df[df['status'] == 'Open'].shape[0]
                       df.groupby('agent_id')['customer_rating'].mean().idxmin()
                    3. Handle cases carefully where values might be missing (like customer_rating or resolution_time_hrs for unresolved tickets).
                    4. PANDAS VERSION RULE: Always use 'df['created_at'].dt.isocalendar().week' instead of '.dt.week' when working with week extractions to avoid deprecation attribute errors.
                    """
                    
                    # Call the Groq API model
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": dataframe_schema_context},
                            {"role": "user", "content": f"Translate this query to pandas code execution syntax: {user_query}"}
                        ],
                        model="llama-3.3-70b-versatile",
                        temperature=0.0,  # Keeping temperature at 0 ensures stable, deterministic code output
                    )
                    
                    # Extract code from response
                    pandas_code_string = chat_completion.choices[0].message.content.strip()
                    
                    # Clean up code output string if the model included stray markdown backticks
                    if pandas_code_string.startswith("```"):
                        pandas_code_string = pandas_code_string.split("\n")[1]
                    if pandas_code_string.endswith("```"):
                        pandas_code_string = pandas_code_string.rsplit("\n", 1)[0]
                    pandas_code_string = pandas_code_string.replace("```python", "").replace("```", "").strip()

                    # Render execution details for evaluation transparency
                    st.info(f"**Generated Backend Logic:** `{pandas_code_string}`")
                    
                    # Safely execute the generated expression against our active dataframe
                    execution_result = eval(pandas_code_string, {"df": df, "pd": pd})
                    
                    # Display the outcome beautifully
                    st.subheader("Analysis Result")
                    if isinstance(execution_result, pd.DataFrame):
                        st.dataframe(execution_result, use_container_width=True)
                    elif isinstance(execution_result, pd.Series):
                        st.dataframe(execution_result)
                    else:
                        st.write(execution_result)
                        
                except Exception as e:
                    st.error(f"An execution or evaluation tracking error occurred: {e}")
                    st.info("Tip: Try rephrasing your query slightly to help the model pick the correct target column parameters.")
