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
import logging
import time
import pytz

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

# Configure logging
logging.basicConfig(filename='digest.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
        api_key = GROQ_API_KEY or "input your Api key here"
        groq_client = Groq(api_key=api_key)
        GROQ_INITIALIZED = True
    except Exception as e:
        logging.error(f"Error initializing Groq client: {e}")
        GROQ_INITIALIZED = False
else:
    GROQ_INITIALIZED = False

# Data loading with file upload support
def load_data():
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Upload Car Sales Data (CSV/Excel)**")
            sales_file = st.file_uploader("Choose file", type=["csv", "xlsx"], key="sales")
        
        if sales_file:
            try:
                # Read file based on extension
                if sales_file.name.endswith('.csv'):
                    data = pd.read_csv(sales_file)
                else:
                    data = pd.read_excel(sales_file)
                
                # Validate required columns
                required_cols = {'year', 'make', 'model', 'trim', 'body', 'transmission', 'vin', 'state', 'condition', 'odometer', 'color', 'interior', 'seller', 'mmr', 'sellingprice', 'saledate'}
                if not all(col in data.columns for col in required_cols):
                    st.error("Data must contain the following columns: " + ", ".join(required_cols))
                    return None
                
                # Calculate KPIs
                data['profit'] = data['sellingprice'] - data['mmr']
                data['profit_margin'] = (data['profit'] / data['mmr']) * 100
                data['saledate'] = pd.to_datetime(data['saledate'], errors='coerce')
                
                # Store in session state
                st.session_state.data = data
                logging.info("Data uploaded successfully")
                return data
            except Exception as e:
                st.error(f"Error processing file: {e}")
                logging.error(f"Error processing file: {e}")
                return None
        else:
            # Fallback to simulated data
            data = pd.DataFrame({
                'year': [2015, 2015, 2014, 2015],
                'make': ['Kia', 'Kia', 'BMW', 'Volvo'],
                'model': ['Sorento', 'Sorento', '3 Series', 'S60'],
                'trim': ['LX', 'LX', '328i SULEV', 'T5'],
                'body': ['SUV', 'SUV', 'Sedan', 'Sedan'],
                'transmission': ['automatic', 'automatic', 'automatic', 'automatic'],
                'vin': ['5xyktca69fg566472', '5xyktca69fg561319', 'wba3c1c51ek116351', 'yv1612tb4f1310987'],
                'state': ['ca', 'ca', 'ca', 'ca'],
                'condition': [5, 5, 45, 41],
                'odometer': [16639, 9393, 1331, 14282],
                'color': ['white', 'white', 'gray', 'white'],
                'interior': ['black', 'beige', 'black', 'black'],
                'seller': ['kia motors america inc', 'kia motors america inc', 'financial services remarketing (lease)', 'volvo na rep/world omni'],
                'mmr': [20500, 20800, 31900, 27500],
                'sellingprice': [21500, 21500, 30000, 27750],
                'saledate': ['2014-12-16', '2014-12-16', '2015-01-15', '2015-01-29']
            })
            data['profit'] = data['sellingprice'] - data['mmr']
            data['profit_margin'] = (data['profit'] / data['mmr']) * 100
            data['saledate'] = pd.to_datetime(data['saledate'], errors='coerce')
            st.session_state.data = data
            logging.info("Loaded fallback simulated data")
            return data

# Store data in SQLite
def store_data(data):
    if data is not None:
        try:
            conn = sqlite3.connect('bi_data.db')
            data.to_sql('car_sales', conn, if_exists='replace', index=False)
            conn.close()
            logging.info("Data stored in SQLite successfully")
        except Exception as e:
            logging.error(f"Error storing data in SQLite: {e}")

# Generate dashboard charts
def generate_dashboard(data):
    if data is not None:
        with st.container():
            st.write("**Car Sales Dashboard**")
            col1, col2 = st.columns(2)
            
            with col1:
                profit_by_make = data.groupby('make')['profit'].sum().reset_index()
                fig_profit = px.bar(profit_by_make, x='make', y='profit', title='Total Profit by Make',
                                    labels={'profit': 'Profit ($)'}, color='make')
                st.plotly_chart(fig_profit, use_container_width=True)
                
                sales_by_body = data['body'].value_counts().reset_index()
                sales_by_body.columns = ['body', 'count']
                fig_body = px.bar(sales_by_body, x='body', y='count', title='Sales Count by Body Type',
                                  labels={'count': 'Number of Sales'}, color='body')
                st.plotly_chart(fig_body, use_container_width=True)
            
            with col2:
                fig_price = px.pie(data, names='make', values='sellingprice',
                                   title='Selling Price Distribution by Make')
                st.plotly_chart(fig_price, use_container_width=True)
                
                profit_margin_by_condition = data.groupby('condition')['profit_margin'].mean().reset_index()
                fig_condition = px.bar(profit_margin_by_condition, x='condition', y='profit_margin',
                                       title='Average Profit Margin by Condition',
                                       labels={'profit_margin': 'Profit Margin (%)'}, color='condition')
                st.plotly_chart(fig_condition, use_container_width=True)

# Generate chart and explanation for KPI query
def generate_kpi_chart_and_explanation(question, data):
    if data is None:
        return None, "Error: No valid data available."
    
    data_summary = "Car Sales Data Summary:\n"
    for make in data['make'].unique():
        make_data = data[data['make'] == make]
        avg_profit = make_data['profit'].mean()
        avg_margin = make_data['profit_margin'].mean()
        sales_count = len(make_data)
        data_summary += f"- {make}: {sales_count} sales, Avg Profit ${avg_profit:.2f}, Avg Profit Margin {avg_margin:.2f}%\n"
    
    fig = None
    x_column = None
    y_column = None
    y_label = None
    title = None
    chart_type = 'bar'
    
    question_lower = question.lower()
    if 'best selling' in question_lower or 'sales' in question_lower:
        if 'model' in question_lower:
            df = data['model'].value_counts().reset_index()
            df.columns = ['model', 'count']
            x_column = 'model'
            y_column = 'count'
            y_label = 'Number of Sales'
            title = 'Sales by Model'
        else:
            df = data['make'].value_counts().reset_index()
            df.columns = ['make', 'count']
            x_column = 'make'
            y_column = 'count'
            y_label = 'Number of Sales'
            title = 'Sales by Make'
    elif 'profit margin' in question_lower:
        df = data.groupby('make')['profit_margin'].mean().reset_index()
        x_column = 'make'
        y_column = 'profit_margin'
        y_label = 'Profit Margin (%)'
        title = 'Average Profit Margin by Make'
    elif 'profit' in question_lower:
        df = data.groupby('make')['profit'].sum().reset_index()
        x_column = 'make'
        y_column = 'profit'
        y_label = 'Profit ($)'
        title = 'Profit by Make'
    elif 'selling price' in question_lower:
        df = data.groupby('make')['sellingprice'].sum().reset_index()
        x_column = 'make'
        y_column = 'sellingprice'
        y_label = 'Selling Price ($)'
        title = 'Selling Price by Make'
    
    if x_column and y_column:
        fig = px.bar(df, x=x_column, y=y_column, title=title,
                     labels={y_column: y_label}, color=x_column)
    
    prompt = f"""
    You are a Business Intelligence Assistant analyzing car sales data. Based on the following data, answer the question: "{question}"
    If a chart is relevant, provide a brief explanation (2-3 sentences) of the chart showing {y_label or 'relevant metrics'} by {x_column or 'grouping'}.
    If no chart is relevant, explain why and provide a text-based answer.
    Data:
    {data_summary}
    Format the response as:
    **Answer**: [Your answer]
    **Chart Explanation**: [Your explanation or reason for no chart]
    """
    
    if GROQ_INITIALIZED:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            explanation = response.choices[0].message.content
        except Exception as e:
            explanation = f"Error: Failed to generate response with Groq: {e}"
            logging.error(explanation)
    else:
        explanation = "Error: Groq AI is not initialized."
    
    return fig, explanation

# Generate weekly digest using Groq
def generate_weekly_digest():
    if not GROQ_INITIALIZED:
        error_msg = "Error: Groq AI is not properly initialized. Please check your API key."
        logging.error(error_msg)
        return error_msg
    
    data = st.session_state.get('data', None)
    if data is None:
        error_msg = "Error: No valid data available for digest generation."
        logging.error(error_msg)
        return error_msg
    
    try:
        store_data(data)
        
        data_summary = "Car Sales Data Summary:\n"
        total_sales = len(data)
        total_profit = data['profit'].sum()
        top_make = data.groupby('make')['profit'].sum().idxmax()
        top_make_profit = data.groupby('make')['profit'].sum().max()
        top_model = data['model'].value_counts().idxmax()
        top_model_sales = data['model'].value_counts().max()
        
        data_summary += f"- Total Sales: {total_sales}\n"
        data_summary += f"- Total Profit: ${total_profit:,.2f}\n"
        data_summary += f"- Top Make by Profit: {top_make} (${top_make_profit:,.2f})\n"
        data_summary += f"- Top Model by Sales: {top_model} ({top_model_sales} sales)\n"
        for make in data['make'].unique():
            make_data = data[data['make'] == make]
            avg_profit = make_data['profit'].mean()
            avg_margin = make_data['profit_margin'].mean()
            sales_count = len(make_data)
            data_summary += f"- {make}: {sales_count} sales, Avg Profit ${avg_profit:.2f}, Avg Profit Margin {avg_margin:.2f}%\n"
        
        prompt = f"""
        You are a Business Intelligence Assistant. Summarize the following car sales data into a concise weekly digest for executives, highlighting the top-performing make by profit, top-selling model, and key insights.
        Data:
        {data_summary}
        Format the response as a professional report with a title and bullet points.
        """
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        report = response.choices[0].message.content
        logging.info("Weekly digest generated successfully")
        return report
    except Exception as e:
        error_msg = f"Error generating digest: {e}"
        logging.error(error_msg)
        return error_msg

# Send report via Email
def send_email_report(report):
    if "Error" in report:
        logging.error(f"Cannot send email: {report}")
        return False
    
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = f"Weekly Car Sales Digest - {datetime.now(pytz.timezone('Africa/Nairobi')).strftime('%Y-%m-%d')}"
    msg.attach(MIMEText(report, 'plain'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        logging.info("Email report sent successfully")
        if 'last_digest_time' not in st.session_state:
            st.session_state.last_digest_time = []
        st.session_state.last_digest_time.append(datetime.now(pytz.timezone('Africa/Nairobi')).strftime('%Y-%m-%d %H:%M:%S %Z'))
        return True
    except Exception as e:
        logging.error(f"Error sending email report: {e}")
        return False

# Background scheduler for weekly digest
def run_scheduler():
    if not SCHEDULER_AVAILABLE:
        logging.error("Scheduler is not available. Install 'schedule' package to enable this feature.")
        return
    
    # Set time zone to EAT
    eat = pytz.timezone('Africa/Nairobi')
    
    def job():
        logging.info("Running scheduled digest job")
        report = generate_weekly_digest()
        if send_email_report(report):
            logging.info("Scheduled email report sent successfully")
        else:
            logging.error("Scheduled email report failed")
    
    schedule.every().monday.at("09:00").do(job).timezone = eat
    logging.info("Scheduler started with EAT timezone")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# Streamlit UI
def main():
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .stApp { max-width: 1200px; margin: 0 auto; }
        .stTabs { margin-top: 20px; }
        .stButton > button { width: 100%; }
        .stFileUploader { margin-bottom: 20px; }
        .stDataFrame { margin-top: 20px; }
        .stSpinner { margin: 20px 0; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Car Sales Business Intelligence Agent")
    
    if not SCHEDULER_AVAILABLE:
        st.warning("Scheduler is not available. Install 'schedule' package to enable automated reports.")
    
    if not GROQ_INITIALIZED:
        st.error("Groq AI is not properly initialized. Please check your API key configuration.")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Dashboard", "Chat", "Weekly Digest"])
    
    # Tab 1: Data Upload
    with tab1:
        st.write("**Upload and Preview Car Sales Data**")
        data = load_data()
        if data is not None:
            with st.container():
                st.write("**Data Preview**")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(data, use_container_width=True)
                with col2:
                    st.write("**Summary**")
                    st.write(f"Total Sales: {len(data)}")
                    st.write(f"Total Profit: ${data['profit'].sum():,.2f}")
                    st.write(f"Average Profit Margin: {data['profit_margin'].mean():.2f}%")
    
    # Tab 2: Dashboard
    with tab2:
        st.write("**Generate Car Sales Dashboard**")
        if st.button("Generate Dashboard", key="dashboard_button"):
            if st.session_state.get('data', None) is not None:
                with st.spinner("Generating dashboard..."):
                    generate_dashboard(st.session_state.data)
            else:
                st.error("No valid data available. Please upload data in the 'Data Upload' tab.")
    
    # Tab 3: Chat Interface
    with tab3:
        st.write("**Chat with BI Agent**")
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        with st.container():
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if message["role"] == "assistant" and "chart" in message and message["chart"]:
                        st.plotly_chart(message["chart"], use_container_width=True)
        
        question = st.chat_input("Ask a KPI question (e.g., 'What’s the average profit for BMW?' or 'Show sales by body type')")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            
            with st.spinner("Processing..."):
                fig, explanation = generate_kpi_chart_and_explanation(question, st.session_state.get('data', None))
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": explanation,
                    "chart": fig
                })
                with st.chat_message("assistant"):
                    st.write(explanation)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Weekly Digest
    with tab4:
        st.write("**Weekly Digest**")
        st.write("The digest is scheduled to send every Monday at 9:00 AM EAT.")
        
        # Display scheduler status
        if SCHEDULER_AVAILABLE:
            st.write("**Scheduler Status**: Running")
            if 'last_digest_time' in st.session_state and st.session_state.last_digest_time:
                st.write("**Last Digest Sent**:")
                for dt in st.session_state.last_digest_time[-3:]:  # Show last 3 digests
                    st.write(f"- {dt}")
            else:
                st.write("**Last Digest Sent**: None")
        else:
            st.error("Scheduler is not available. Install 'schedule' package to enable automated digests.")
        
        # Manual digest trigger
        if st.button("Send Weekly Digest Now", key="digest_button"):
            with st.spinner("Generating and sending digest..."):
                report = generate_weekly_digest()
                if send_email_report(report):
                    st.success("Weekly digest sent successfully!")
                else:
                    st.error("Failed to send weekly digest. Check logs (digest.log) for details.")
        
        # Display recent log entries
        st.write("**Recent Log Entries**")
        try:
            with open('digest.log', 'r') as f:
                logs = f.readlines()
                for log in logs[-5:]:  # Show last 5 log entries
                    st.write(log.strip())
        except FileNotFoundError:
            st.write("No logs available. Try sending a digest to generate logs.")

if __name__ == "__main__":
    # Start scheduler in a separate thread only if available
    if SCHEDULER_AVAILABLE:
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    # Run Streamlit app
    main()