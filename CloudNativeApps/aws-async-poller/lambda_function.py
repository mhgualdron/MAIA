import json
import os
import time
import datetime
import requests
import psycopg2
import boto3

DB_URI = os.getenv("DB_URI", "postgresql://admin:admin@localhost:5432/credit_cards_db")
TRUENATIVE_HOST = os.getenv("TRUENATIVE_HOST", "http://truenative.default.svc.cluster.local")
TRUENATIVE_SECRET = os.getenv("TRUENATIVE_SECRET", "")
SES_SENDER_EMAIL = os.getenv("SES_SENDER_EMAIL", "noreply@midominio.com")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

def update_card_status(card_id: str, new_status: str):
    """Update status conditionally via psycopg2 directly to reduce Lambda's memory footprint."""
    conn = None
    try:
        conn = psycopg2.connect(DB_URI)
        cur = conn.cursor()
        now = datetime.datetime.utcnow()
        print(f"Executing update for card_id={card_id} to status={new_status}")
        cur.execute(
            """
            UPDATE credit_cards 
            SET status = %s, "updatedAt" = %s 
            WHERE id = %s
            """, 
            (new_status, now, card_id)
        )
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"Error updating database: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def get_truenative_status(ruv: str) -> str:
    """Invoca la API de truenative simulando polling asincrono externo"""
    headers = {"Authorization": f"Bearer {TRUENATIVE_SECRET}"}
    try:
        url = f"{TRUENATIVE_HOST}/native/cards/{ruv}"
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("status", "POR_VERIFICAR").upper()
        else:
            print(f"Error from TrueNative: {r.status_code}")
            return "POR_VERIFICAR"
    except Exception as e:
        print(f"Http request error TrueNative: {e}")
        return "POR_VERIFICAR"

def send_notification_email(user_email: str, ruv: str, status: str):
    if not SES_SENDER_EMAIL or SES_SENDER_EMAIL == "noreply@midominio.com":
        print("Sender SES omitido, asumiendo ambiente dev/local o falta config")
        return
        
    try:
        client = boto3.client("ses", region_name=AWS_REGION)
        message = (
            f"Estimado usuario,\n\n"
            f"El estado de verificacion de tu nueva tarjeta ha finalizado.\n"
            f"Estado final: {status}\n"
            f"RUV asociado: {ruv}\n\n"
            f"El equipo de Cloud Apps"
        )
        
        response = client.send_email(
            Destination={
                "ToAddresses": [user_email]
            },
            Message={
                "Body": {
                    "Text": {
                        "Charset": "UTF-8",
                        "Data": message
                    }
                },
                "Subject": {
                    "Charset": "UTF-8",
                    "Data": "[Cloud Apps] Actualizacion resultado verificacion Tarjeta"
                }
            },
            Source=SES_SENDER_EMAIL
        )
        print(f"Email sent, MessageId: {response['MessageId']}")
    except Exception as e:
        print(f"Error sending SES email: {e}")

def lambda_handler(event, context):
    print("Iniciando procesamiento lambda", event)
    
    for record in event.get('Records', []):
        try:
            body = json.loads(record.get('body', "{}"))
            
            card_id = body.get("cardId")
            ruv = body.get("ruv")
            email = body.get("userEmail")
            created_str = body.get("createdAt")
            
            if not card_id or not ruv:
                print("Mensaje invalido: " + str(body))
                continue
                
            # Calcular antiguedad para forzar abort a los ~30 seg
            # createdAt llega del pydantic como `isoformat()`
            try:
                created_at = datetime.datetime.fromisoformat(created_str)
            except Exception:
                created_at = datetime.datetime.utcnow()

            final_status = "POR_VERIFICAR"
            timeout_seconds = 30
            
            loop_duration = 0
            poll_interval = 1
            
            while loop_duration < timeout_seconds:
                # Verificamos si pasaron 30s desde que se inserto
                elapsed_since_creation = (datetime.datetime.utcnow() - created_at).total_seconds()
                
                if elapsed_since_creation >= timeout_seconds:
                    print("Timeout absoluto desde insercion web excedido (30s).")
                    final_status = "RECHAZADA"
                    break
                    
                status_tn = get_truenative_status(ruv)
                print(f"Poll result for RUV={ruv}: {status_tn}")
                
                if status_tn in ["APROBADA", "RECHAZADA"]:
                    final_status = status_tn
                    break
                    
                time.sleep(poll_interval)
                loop_duration += poll_interval
            
            # Use the loop_duration to track if we timed out
            if loop_duration >= timeout_seconds and final_status == "POR_VERIFICAR":
                print(f"Polling timed out for RUV={ruv}. Forcing RECHAZADA.")
                final_status = "RECHAZADA"
            
            print(f"Resolved RUV={ruv} to status={final_status}")
            update_card_status(card_id, final_status)
            
            try:
                send_notification_email(email, ruv, final_status)
            except Exception as e:
                print(f"Non-blocking error sending email: {e}")

        except Exception as msg_err:
            print(f"Error processing record: {msg_err}")
            
    return {
        "statusCode": 200,
        "body": "Processed gracefully"
    }
