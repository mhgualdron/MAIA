import bcrypt
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from app.models import User

def hash_password(password: str) -> tuple[str, str]:
    """
    Retorna (password_hash, salt)
    """
    # Generar salt
    salt = bcrypt.gensalt().decode('utf-8')
    # Hashear password con el salt
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt.encode('utf-8')).decode('utf-8')
    return hashed, salt

def verify_password(plain_password: str, hashed_password: str, salt: str) -> bool:
    """
    Verifica que el password plano coincida con el hash usando el salt dado
    """
    # Se debe regenerar el hash con el password plano y el salt ORIGINAL
    check_hash = bcrypt.hashpw(plain_password.encode('utf-8'), salt.encode('utf-8')).decode('utf-8')
    return check_hash == hashed_password
