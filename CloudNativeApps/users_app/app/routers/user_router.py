from fastapi import APIRouter, Depends, status, BackgroundTasks, Request, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import UserCreate, UserResponse, UserLogin, Token, UserUpdate, UserVerificationCallback
from app.repositories.user_repository import UserRepository
from app.services.user_service import UserService
from app.models import User
from app.auth import get_current_user

# Creación del router
router = APIRouter(prefix="/users", tags=["users"])

# Dependencias
def get_user_service(db: Session = Depends(get_db)) -> UserService:
    repository = UserRepository(db)
    return UserService(repository)

# --- Endpoints ---

# Health check
@router.get("/ping")
def health_check():
    return "pong"

# Resetear la ddbb
@router.post("/reset")
def reset_db(service: UserService = Depends(get_user_service)):
    service.reset_db()
    return {"msg": "Todos los datos fueron eliminados"}

# Crear un usuario

@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, background_tasks: BackgroundTasks, service: UserService = Depends(get_user_service)):
    new_user = service.create_user(user)
    
    # Mover la llamada a SNS a segundo plano para evitar Timeouts (502)
    background_tasks.add_task(
        service.sns_service.publish_user_created,
        user_id=new_user.id,
        email=new_user.email,
        dni=new_user.dni,
        full_name=new_user.fullName,
        phone=new_user.phoneNumber
    )
    
    return {
        "id": new_user.id,
        "createdAt": new_user.createdAt.isoformat()
    }

# Login
@router.post("/auth", response_model=Token)
def login(credentials: UserLogin, service: UserService = Depends(get_user_service)):
    user = service.login_user(credentials)
    return {
        "id": user.id,
        "token": user.token,
        "expireAt": user.expireAt
    }

# Obtener el usuario actual
@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# Actualizar un usuario
@router.patch("/{id}")
def update_user(id: str, user_update: UserUpdate, service: UserService = Depends(get_user_service)):
    service.update_user(id, user_update)
    return {"msg": "el usuario ha sido actualizado"}

# Webhook de TrueNative para verificación de identidad (RF-007)
@router.post("/verify/callback")
@router.patch("/verify/callback")
async def verification_callback(request: Request, service: UserService = Depends(get_user_service)):
    import logging
    logger = logging.getLogger("uvicorn.error")
    try:
        raw_body = await request.body()
        logger.warning(f"TRUE NATIVE RAW BODY: {raw_body}")
        
        data = await request.json()
        logger.warning(f"TRUE NATIVE JSON: {data}")
        
        callback_data = UserVerificationCallback(**data)
        service.process_verification_callback(callback_data)
        return {"msg": "Callback procesado"}
    except Exception as e:
        import traceback
        logger.error(f"TRUE NATIVE WEBHOOK ERROR: {traceback.format_exc()}")
        return {"msg": "Error en callback procesado"}

# Contar usuarios

@router.get("/count")
def count_users(service: UserService = Depends(get_user_service)):
    count = service.get_count()
    return {"count": count}
