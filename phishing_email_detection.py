import streamlit as st
import re
import nltk
import imaplib
import email
from email.header import decode_header
from datetime import datetime
import joblib
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load the pre-trained model
model_path = 'D:/Uni/7th_sem/NLP/Project/models/email_detection_model4.pkl' #change the path
classifier = joblib.load(model_path)

# Ensure stopwords are downloaded only once
def remove_stopwords(body):
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')  # Download if not found

    stop_words = set(stopwords.words('english'))
    words = body.split()

    return " ".join(words)

# Function to clean HTML tags and normalize symbols (with lowercase conversion)
def clean_html_body(body):
    soup = BeautifulSoup(body, "html.parser")
    text = soup.get_text()

    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.strip()  # Convert to lowercase and strip leading/trailing spaces

    return text

# Function to preprocess email body (clean and remove stopwords)
def preprocess_email_body(body):
    clean_body = clean_html_body(body)
    processed_body = remove_stopwords(clean_body)
    return processed_body

# Function to classify the email
def classify_email(email_text):
    prediction = classifier.predict([email_text])  # Classify the email
    if prediction == 1:
        return "Safe Email"
    else:
        return "Phishing Email"

# Function to perform sentiment analysis
def analyze_sentiment(email_text):
    scores = sia.polarity_scores(email_text)
    compound = scores['compound']
    if compound > 0.05:
        return "Positive", "green"
    elif -0.05 <= compound <= 0.05:
        return "Neutral", "orange"
    else:
        return "Negative", "red"

# Function to fetch emails
def fetch_emails_from_gmail(email_user, email_password, max_emails, folder):
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_user, email_password)

        mail.select(f'"{folder}"')
        status, messages = mail.search(None, "ALL")

        email_ids = messages[0].split()
        email_ids = email_ids[::-1]  # Reverse to get the latest emails first
        email_ids = email_ids[:max_emails+1]  # Process the top 'max_emails' items


        emails_data = []
        phishing_count = 0
        safe_count = 0

        for email_id in email_ids:
            status, msg_data = mail.fetch(email_id, "(RFC822)")

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])

                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or "utf-8") if encoding else subject.decode()

                    if not subject:
                        subject = "No Subject"

                    body = None
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition"))
                            if "attachment" not in content_disposition and content_type == "text/plain":
                                try:
                                    body = part.get_payload(decode=True).decode(errors="ignore")
                                    if body:
                                        break
                                except Exception as e:
                                    print(f"Error decoding body part: {e}")
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode(errors="ignore")
                        except Exception as e:
                            print(f"Error decoding email body: {e}")

                    if not body:
                        continue

                    processed_body = preprocess_email_body(body)

                    classification_result = classify_email(subject + " " + processed_body)

                    if classification_result == "Phishing Email":
                        phishing_count += 1
                    else:
                        safe_count += 1

                    sentiment_result, sentiment_color = analyze_sentiment(processed_body)

                    emails_data.append({
                        "subject": subject,
                        "body": processed_body,
                        "classification": classification_result,
                        "sentiment": sentiment_result,
                        "sentiment_color": sentiment_color
                    })

        return emails_data, phishing_count+safe_count, phishing_count, safe_count

    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return [], 0, 0, 0

# Streamlit Interface
def main():
    st.title("Phishing Emails Detection System")
    st.sidebar.write("#### Natural Language Processing Project | BSE-7(B)")
    st.sidebar.info("Ensure that 2FA is enabled in your Gmail account and a custom app password is created for this application.")
    st.sidebar.title("Settings")

    email_user = st.sidebar.text_input("Email Address", type="default")
    email_password = st.sidebar.text_input("App Password", type="password")
    max_emails = st.sidebar.slider("Max Emails to Fetch", min_value=1, max_value=50, value=5)

    folder = st.sidebar.segmented_control("Select Folder", ["Inbox", "Spam", "All Mail"])
    folder_map = {"Inbox": "INBOX", "Spam": "[Gmail]/Spam", "All Mail": "[Gmail]/All Mail"}
    folderm = folder_map[folder]

    start_fetch = st.sidebar.button("Fetch Emails")

    if start_fetch:
        if not email_user or not email_password:
            st.error("Please provide valid email credentials!")
        else:
            with st.spinner("Fetching and classifying emails..."):
                emails, total_emails, phishing_count, safe_count = fetch_emails_from_gmail(email_user, email_password, max_emails, folderm)
                col1, col2, col3 = st.columns(3)
                # Initialize session state for previous counts if not already set
                if "previous_total_emails" not in st.session_state:
                    st.session_state["previous_total_emails"] = 0
                if "previous_phishing_emails" not in st.session_state:
                    st.session_state["previous_phishing_emails"] = 0
                if "previous_safe_emails" not in st.session_state:
                    st.session_state["previous_safe_emails"] = 0

                # Calculate differences for metrics
                total_diff = total_emails - st.session_state["previous_total_emails"]
                phishing_diff = phishing_count - st.session_state["previous_phishing_emails"]
                safe_diff = safe_count - st.session_state["previous_safe_emails"]

                # Display metrics
                with col1:
                    st.metric("Total Emails", total_emails, f"{total_diff:+}", border=True)
                with col2:
                    st.metric("Phishing/Spam Emails 🚨", phishing_count, f"{phishing_diff:+}", delta_color="inverse", border=True)
                with col3:
                    st.metric("Safe Emails ✅", safe_count, f"{safe_diff:+}", border=True)

                # Update session state with current counts
                st.session_state["previous_total_emails"] = total_emails
                st.session_state["previous_phishing_emails"] = phishing_count
                st.session_state["previous_safe_emails"] = safe_count


                if emails:
                    idx = 1
                    st.success(f"Fetched and classified {total_emails} emails from {folder}!")
                    for email in emails:
                        st.write(f"#### Email {idx}")
                        with st.container():
                            idx+=1
                            st.markdown(
                                f"""
                                <div style='border-radius: 10px; padding: 15px; margin-bottom: 15px; border: 1px solid #444; padding-bottom:5px;'>
                                    <p style='color: #f1f1f1; margin: 0; font-size: 18px'><strong>Subject:</strong> {email['subject']}</p>
                                    <p style='color: #f1f1f1; margin: 0;'>
                                        <strong>Sentiment:</strong> {email['sentiment']}
                                        <span style='width: 10px; height: 10px; background-color: {email['sentiment_color']}; display: inline-block; border-radius: 50%; margin-left: 5px;'></span>
                                    </p>
                                    {"<p style='color: #4caf50;'><b>Safe Email ✅</b></p>" if email['classification'] == "Safe Email" else "<p style='color: #f44336;'><b>Potential Phishing/Spam Email 🚨</b></p>"}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            with st.expander("See Email Body"):
                                st.write(email["body"])
                else:
                    st.warning("No emails were fetched or classified.")

if __name__ == "__main__":
    main()
