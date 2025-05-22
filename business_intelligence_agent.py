import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os
import streamlit as st
import threading
import plotly.express as px

# Optional imports with error handling
try:
    import schedule
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    print("Warning: 'schedule' package not found. Automated scheduling will be disabled.")
    print("To enable scheduling, install it using: pip install schedule")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: 'groq' package not found. AI features will be disabled.")
    print("To enable AI features, install it using: pip install groq")

# Load environment variables
load_dotenv()

# Email and Groq configuration
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client with error handling
groq_client = None
if GROQ_AVAILABLE:
    try:
        api_key = GROQ_API_KEY or "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"
        groq_client = Groq(api_key=api_key)
        GROQ_INITIALIZED = True
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        GROQ_INITIALIZED = False
else:
    GROQ_INITIALIZED = False

# Data loading with file upload support
def load_data():
    st.subheader("Upload Data Files")
    marketing_file = st.file_uploader("Upload Marketing Data (CSV/Excel)", type=["csv", "xlsx"])
    sales_file = st.file_uploader("Upload Sales Data (CSV/Excel)", type=["csv", "xlsx"])
    
    if marketing_file and sales_file:
        try:
            # Read files based on extension
            if marketing_file.name.endswith('.csv'):
                marketing_data = pd.read_csv(marketing_file)
            else:
                marketing_data = pd.read_excel(marketing_file)
                
            if sales_file.name.endswith('.csv'):
                sales_data = pd.read_csv(sales_file)
            else:
                sales_data = pd.read_excel(sales_file)
            
            # Validate required columns
            required_cols = {'channel', 'cost', 'impressions'}
            if not all(col in marketing_data.columns for col in required_cols):
                st.error("Marketing data must contain 'channel', 'cost', and 'impressions' columns.")
                return None
            if 'channel' not in sales_data.columns or 'revenue' not in sales_data.columns:
                st.error("Sales data must contain 'channel' and 'revenue' columns.")
                return None
            
            # Merge data
            data = pd.merge(marketing_data, sales_data, on='channel')
            data['roi'] = (data['revenue'] - data['cost']) / data['cost'] * 100
            return data
        except Exception as e:
            st.error(f"Error processing files: {e}")
            return None
    else:
        # Fallback to simulated data
        marketing_data = pd.DataFrame({
            'channel': ['Google Ads', 'Facebook Ads', 'Email'],
            'cost': [5000, 3000, 1000],
            'impressions': [100000, 75000, 20000]
        })
        sales_data = pd.DataFrame({
            'channel': ['Google Ads', 'Facebook Ads', 'Email'],
            'revenue': [15000, 9000, 4000]
        })
        data = pd.merge(marketing_data, sales_data, on='channel')
        data['roi'] = (data['revenue'] - data['cost']) / data['cost'] * 100
        return data

# Store data in SQLite
def store_data(data):
    if data is not None:
        conn = sqlite3.connect('bi_data.db')
        data.to_sql('performance', conn, if_exists='replace', index=False)
        conn.close()

# Generate charts
def generate_charts(data):
    if data is not None:
        # ROI Bar Chart
        fig_bar = px.bar(data, x='channel', y='roi', title='ROI by Channel', 
                         labels={'roi': 'ROI (%)'}, color='channel')
        st.plotly_chart(fig_bar)
        
        # Revenue Pie Chart
        fig_pie = px.pie(data, names='channel', values='revenue', title='Revenue Distribution by Channel')
        st.plotly_chart(fig_pie)

# Generate weekly digest using Groq
def generate_weekly_digest():
    if not GROQ_INITIALIZED:
        return "Error: Groq AI is not properly initialized. Please check your API key."
    
    data = load_data()
    if data is None:
        return "Error: No valid data available for digest generation."
    
    store_data(data)
    
    data_summary = "Channel Performance Data:\n"
    for _, row in data.iterrows():
        data_summary += f"- {row['channel']}: Revenue ${row['revenue']}, Cost ${row['cost']}, ROI {row['roi']:.2f}%\n"
    
    prompt = f"""
    You are a Business Intelligence Assistant. Summarize the following data into a concise weekly digest for executives, highlighting the highest ROI channel and key insights.
    Data:
    {data_summary}
    Format the response as a professional report with a title and bullet points.
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Send report via Email
def send_email_report(report):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = f"Weekly BI Digest - {datetime.now().strftime('%Y-%m-%d')}"
    msg.attach(MIMEText(report, 'plain'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        return True
    except Exception as e:
        print(f"Error sending email report: {e}")
        return False

# Answer specific questions using Groq
def answer_question(question, data):
    if not GROQ_INITIALIZED:
        return "Error: Groq AI is not properly initialized. Please check your API key."
    
    if data is None:
        return "Error: No valid data available to answer the question."
    
    data_summary = "Channel Performance Data:\n"
    for _, row in data.iterrows():
        data_summary += f"- {row['channel']}: Revenue ${row['revenue']}, Cost ${row['cost']}, ROI {row['roi']:.2f}%\n"
    
    prompt = f"""
    You are a Business Intelligence Assistant. Using the following data, answer the question: "{question}"
    Data:
    {data_summary}
    Provide a concise and accurate response.
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    
    return response.choices[0].message.content

# Background scheduler for weekly digest
def run_scheduler():
    if not SCHEDULER_AVAILABLE:
        print("Scheduler is not available. Install 'schedule' package to enable this feature.")
        return
    
    schedule.every().monday.at("09:00").do(job)
    while True:
        schedule.run_pending()
        time.sleep(60)

def job():
    report = generate_weekly_digest()
    if send_email_report(report):
        print("Scheduled email report sent successfully")

# Streamlit UI
def main():
    st.title("Business Intelligence Agent")
    
    if not SCHEDULER_AVAILABLE:
        st.warning("Scheduler is not available. Install 'schedule' package to enable automated reports.")
    
    if not GROQ_INITIALIZED:
        st.error("Groq AI is not properly initialized. Please check your API key configuration.")
    
    st.write("Upload data, view performance metrics, explore charts, ask questions, and generate reports.")
    
    # Load and display data
    data = load_data()
    if data is not None:
        st.subheader("Channel Performance Data")
        st.dataframe(data)
        
        # Generate and display charts
        st.subheader("Data Visualizations")
        generate_charts(data)
    
    # Chat interface
    st.subheader("Chat with BI Agent")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    question = st.chat_input("Ask a question about the data (e.g., 'Which channel gave us the highest ROI?')")
    if question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        # Get and display assistant response
        with st.spinner("Processing..."):
            answer = answer_question(question, data)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
    
    # Manual digest generation
    st.subheader("Generate Weekly Digest")
    if st.button("Send Weekly Digest Now"):
        with st.spinner("Generating and sending digest..."):
            report = generate_weekly_digest()
            if send_email_report(report):
                st.success("Weekly digest sent successfully!")
            else:
                st.error("Failed to send weekly digest. Check logs for details.")

if __name__ == "__main__":
    # Start scheduler in a separate thread only if available
    if SCHEDULER_AVAILABLE:
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    # Run Streamlit app
    main()