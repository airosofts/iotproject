from flask import Flask, request, jsonify
from supabase import create_client, Client
import re
import smtplib
import dns.resolver
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import concurrent.futures
import uuid
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MAX_WORKERS = 200
SMTP_TIMEOUT = 8
DNS_TIMEOUT = 5
RETRY_ATTEMPTS = 2

DISPOSABLE_DOMAINS = {
    'tempmail.com', '10minutemail.com', 'guerrillamail.com',
    'mailinator.com', 'throwawaymail.com', 'fakeinbox.com'
}

# --------------------------------------------------
# Supabase Setup
# --------------------------------------------------
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://arzjyxgjuxygehnxnvqv.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFyemp5eGdqdXh5Z2VobnhudnF2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjA3MTAwNDYsImV4cCI6MjAzNjI4NjA0Nn0.dLf5ZJxCOZ1VVCxuFrssvJH2B2-fXm_B5KRRFv1mRrs')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------------------------------
# Caching System
# --------------------------------------------------
mx_cache = {}
mx_cache_lock = threading.Lock()
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']
dns.resolver.default_resolver.timeout = DNS_TIMEOUT
dns.resolver.default_resolver.lifetime = DNS_TIMEOUT

# --------------------------------------------------
# Validation Functions
# --------------------------------------------------
def validate_email_format(email):
    if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', email):
        return False
    if email.lower().startswith(('postmaster@', 'abuse@', 'webmaster@')):
        return False
    return True

def is_disposable_domain(domain):
    return domain.lower() in DISPOSABLE_DOMAINS

def get_mx_records(domain):
    with mx_cache_lock:
        if domain in mx_cache:
            return mx_cache[domain]

    try:
        mx_records = []
        answers = dns.resolver.resolve(domain, 'MX')
        sorted_records = sorted([(r.preference, str(r.exchange).rstrip('.')) for r in answers], key=lambda x: x[0])
        mx_records = [record[1] for record in sorted_records]
        
        with mx_cache_lock:
            mx_cache[domain] = mx_records
        return mx_records
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers):
        with mx_cache_lock:
            mx_cache[domain] = None
        return None
    except Exception as e:
        print(f"DNS Error for {domain}: {str(e)}")
        return None

def check_a_record(domain):
    try:
        dns.resolver.resolve(domain, 'A')
        return True
    except:
        return False

def verify_smtp(email, mx_records):
    for attempt in range(RETRY_ATTEMPTS):
        for mx in mx_records:
            try:
                with smtplib.SMTP(timeout=SMTP_TIMEOUT) as server:
                    server.connect(mx)
                    server.helo('example.com')
                    server.mail('noreply@example.com')
                    code, msg = server.rcpt(email)
                    if code == 250:
                        return True, "Mailbox exists"
                    elif code in (450, 451, 452, 503):
                        return False, f"Temporary error: {code} {msg}"
                    else:
                        return False, f"Permanent error: {code} {msg}"
            except smtplib.SMTPServerDisconnected:
                continue
            except Exception as e:
                continue
        time.sleep(1)
    
    return False, "All verification attempts failed"

def validate_email(email):
    if not validate_email_format(email):
        return {'Email': email, 'Valid': False, 'Message': 'Invalid format'}
    
    domain = email.split('@')[1]
    
    if is_disposable_domain(domain):
        return {'Email': email, 'Valid': False, 'Message': 'Disposable domain'}
    
    mx_records = get_mx_records(domain)
    if not mx_records:
        if check_a_record(domain):
            return {'Email': email, 'Valid': False, 'Message': 'No MX but domain exists'}
        return {'Email': email, 'Valid': False, 'Message': 'No valid DNS records'}
    
    valid, message = verify_smtp(email, mx_records)
    return {'Email': email, 'Valid': valid, 'Message': message}

# --------------------------------------------------
# Process Emails
# --------------------------------------------------
def process_emails(emails):
    logger.debug(f"Received {len(emails)} emails: {emails}")
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(validate_email, email): email for email in emails}
        for future in concurrent.futures.as_completed(futures):
            email = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.debug(f"Processed {email}: {result}")
            except Exception as e:
                result = {'Email': email, 'Valid': False, 'Message': str(e)}
                results.append(result)
                logger.error(f"Error processing {email}: {str(e)}")
    
    logger.debug(f"Returning {len(results)} results")
    return results

# --------------------------------------------------
# API Endpoint
# --------------------------------------------------
@app.route('/api/validate-emails', methods=['POST'])
def validate_emails():
    try:
        data = request.get_json()
        logger.debug(f"Raw request data: {data}")
        if not data or 'emails' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing emails in request body',
                'request_id': str(uuid.uuid4())
            }), 400

        emails = data['emails']
        if not isinstance(emails, list):
            return jsonify({
                'status': 'error',
                'message': 'Emails must be provided as a list',
                'request_id': str(uuid.uuid4())
            }), 400

        logger.debug(f"Processing {len(emails)} emails")
        if not emails:
            return jsonify({
                'status': 'success',
                'message': 'No emails provided',
                'results': [],
                'request_id': str(uuid.uuid4())
            }), 200

        start_time = time.time()
        results = process_emails(emails)
        duration = time.time() - start_time
        
        df = pd.DataFrame(results)
        valid_count = df['Valid'].sum() if not df.empty else 0
        total_count = len(df) if not df.empty else 0

        return jsonify({
            'status': 'success',
            'message': f'Processed {total_count} emails',
            'results': results,
            'stats': {
                'total_emails': total_count,
                'valid_emails': int(valid_count),
                'validation_accuracy': (valid_count/total_count*100) if total_count > 0 else 0,
                'processing_time_seconds': round(duration, 2)
            },
            'request_id': str(uuid.uuid4())
        }), 200

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}',
            'request_id': str(uuid.uuid4())
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Use Railway's PORT or default to 8080
    app.run(host='0.0.0.0', port=port, debug=True)