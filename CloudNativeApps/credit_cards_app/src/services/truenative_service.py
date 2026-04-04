import requests
from fastapi import HTTPException
from src.core.config import settings

def register_card(payload: dict) -> dict:
    headers = {"Authorization": f"Bearer {settings.TRUENATIVE_SECRET}"}
    try:
        response = requests.post(f"{settings.TRUENATIVE_HOST}/native/cards", json=payload, headers=headers, timeout=10)
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Timeout TrueNative")
    except Exception:
        raise HTTPException(status_code=502, detail="Error de red con TrueNative")
    
    if response.status_code == 201:
        return response.json()
    else:
        raise HTTPException(status_code=400, detail=f"TrueNative rechazó (Status {response.status_code}): {response.text}")
