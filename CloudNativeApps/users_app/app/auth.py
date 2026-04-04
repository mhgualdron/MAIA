import uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User

from typing import Optional

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/auth", auto_error=False)

def create_access_token() -> str:
    """
    Genera un token opaco (UUID4 random)
    NO ES JWT
    """
    return str(uuid.uuid4())

def get_current_user(token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Valida token contra la BD y retorna usuario
    """
    # Si no hay token, retornar 403 (sin autenticación)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated"
        )
    
    user = db.query(User).filter(User.token == token).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # Validar expiración
    if user.expireAt and user.expireAt < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    return user
