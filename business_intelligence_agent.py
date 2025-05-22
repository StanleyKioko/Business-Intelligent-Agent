import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import slack_sdk
from dotenv import load_dotenv
import os
import schedule
import time
import uuid

# Load environment variables
load_dotenv()

# Slack and Email configuration
SLACK_TOKEN = os.getenv("SLACK_TOKEN")
SLACK_CHANNEL = "#bi-reports"
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = "executive_team@example.com"

# Initialize Slack client
slack_client = slack_sdk.WebClient(token=SLACK_TOKEN)

# Simulated data loading (replace with API calls or database queries)
def load_data():
    # Sample data: marketing, sales, finance
    marketing_data = pd.DataFrame({
        'channel': ['Google Ads', 'Facebook Ads', 'Email'],
        'cost': [5000, 3000, 1000],
        'impressions': [100000, 75000, 20000]
    })
    
    sales_data = pd.DataFrame({
        'channel': ['Google Ads', 'Facebook Ads', 'Email'],
        'revenue': [15000, 9000, 4000]
    })
    
    # Merge data
    data = pd.merge(marketing_data, sales_data, on='channel')
    return data

# Calculate ROI
def calculate_roi(data):
    data['roi'] = (data['revenue'] - data['cost']) / data['cost'] * 100
    return data

# Generate weekly digest
def generate_weekly_digest():
    data = load_data()
    data = calculate_roi(data)
    
    # Create report
    report = f"Weekly Business Intelligence Digest - {datetime.now().strftime('%Y-%m-%d')}\n\n"
    report += f"Highest ROI Channel: {data.loc[data['roi'].idxmax(), 'channel']} ({data['roi'].max():.2f}%)\n\n"
    report += "Channel Performance:\n"
    for _, row in data.iterrows():
        report += f"- {row['channel']}: Revenue ${row['revenue']}, Cost ${row['cost']}, ROI {row['roi']:.2f}%\n"
    
    return report

# Send report via Slack
def send_slack_report(report):
    try:
        slack_client.chat_postMessage(channel=SLACK_CHANNEL, text=report)
        print("Slack report sent successfully")
    except Exception as e:
        print(f"Error sending Slack report: {e}")

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
        print("Email report sent successfully")
    except Exception as e:
        print(f"Error sending email report: {e}")

# Answer specific questions (basic keyword-based approach)
def answer_question(question):
    data = load_data()
    data = calculate_roi(data)
    
    if "highest roi" in question.lower() and "last month" in question.lower():
        highest_roi_channel = data.loc[data['roi'].idxmax(), 'channel']
        highest_roi = data['roi'].max()
        return f"The channel with the highest ROI last month was {highest_roi_channel} with an ROI of {highest_roi:.2f}%."
    return "Sorry, I don't understand the question. Try asking about ROI or channel performance."

# Schedule weekly digest
def job():
    report = generate_weekly_digest()
    send_slack_report(report)
    send_email_report(report)

# Main execution
if __name__ == "__main__":
    # Example question
    question = "Which channel gave us the highest ROI last month?"
    print(answer_question(question))
    
    # Schedule weekly digest (every Monday at 9 AM)
    schedule.every().monday.at("09:00").do(job)
    
    # Run scheduler
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute