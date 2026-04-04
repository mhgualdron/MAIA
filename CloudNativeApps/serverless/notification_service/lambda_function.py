import json
import os
import smtplib
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Credenciales Brevo para Laboratorio (RF-007)
SMTP_USER = "a436a4001@smtp-brevo.com"
SMTP_PASS = "pNnZCFdkAPHScXLR" # API Key de Brevo
SMTP_HOST = "smtp-relay.brevo.com"
SMTP_PORT = 587

def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")
    
    # 1. Parse SNS Message
    try:
        sns_record = event['Records'][0]['Sns']
        sns_message = json.loads(sns_record['Message'])
        
        event_type = sns_message.get('event')
        data = sns_message.get('data')
        
    except Exception as e:
        print(f"Error parsing SNS message: {e}")
        return {"statusCode": 400, "body": "Invalid SNS message"}

    if event_type != "VERIFICATION_COMPLETED":
        print(f"Skipping event type: {event_type}")
        return {"statusCode": 200, "body": "Skipped"}

    # 2. Extract Data
    recipient_email = data.get('email')
    status = data.get('status')
    ruv = data.get('ruv')
    user_id = data.get('user_id') or data.get('userIdentifier')

    if not recipient_email:
        print("Recipient email missing")
        return {"statusCode": 400, "body": "Recipient email missing"}

    # 3. Create Email Content
    subject = f"Resultado de Verificación de Identidad - {status}"
    body_text = (f"Hola,\n\n"
                 f"El proceso de verificación de tu cuenta ha finalizado.\n"
                 f"Estado final: {status}\n"
                 f"RUV: {ruv}\n"
                 f"ID de Usuario: {user_id}\n\n"
                 f"Gracias por usar nuestra plataforma.")
    
    body_html = f"""<html>
<head></head>
<body>
  <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; border: 1px solid #ddd; padding: 20px;">
    <h2 style="color: #2c3e50;">Resultado de tu Verificación</h2>
    <p>Hola,</p>
    <p>Te informamos que el proceso de verificación de identidad ha concluido:</p>
    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px;">
        <p><strong>Estado:</strong> <span style="color: #2980b9;">{status}</span></p>
        <p><strong>RUV:</strong> {ruv}</p>
        <p><strong>ID de Usuario:</strong> {user_id}</p>
    </div>
    <p>Gracias por confiar en <strong>Dann App</strong>.</p>
  </div>
</body>
</html>"""

    # 4. Send Email via SMTP (Brevo)
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"Dann App <j.buriticar2@uniandes.edu.co>"
        msg['To'] = recipient_email

        msg.attach(MIMEText(body_text, 'plain'))
        msg.attach(MIMEText(body_html, 'html'))

        print(f"Connecting to Brevo SMTP for {recipient_email}...")
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
            
        print(f"Email sent successfully to {recipient_email}")
        return {"statusCode": 200, "body": "Email sent"}
    except Exception as e:
        print(f"Error sending email: {traceback.format_exc()}")
        return {"statusCode": 500, "body": str(e)}
