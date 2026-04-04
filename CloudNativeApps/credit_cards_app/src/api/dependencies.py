from fastapi import Request, HTTPException
from src.services.users_service import get_current_user

def auth_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="No hay token o es inválido")
    token = auth_header.split(" ")[1]
    
    user_data = get_current_user(token)
    return user_data
