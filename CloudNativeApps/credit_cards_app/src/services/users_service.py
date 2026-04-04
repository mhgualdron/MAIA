import requests
from fastapi import HTTPException
from src.core.config import settings

def get_current_user(token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(f"{settings.USERS_APP_HOST}/users/me", headers=headers, timeout=5)
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Timeout users_app")
    except Exception:
        raise HTTPException(status_code=502, detail="Error de comunicación con users_app")
        
    if response.status_code == 200:
        return response.json()
    elif response.status_code in (401, 403):
        raise HTTPException(status_code=response.status_code, detail="No autorizado")
    else:
        raise HTTPException(status_code=502, detail="Error en users_app")
