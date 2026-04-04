from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import User
from app.schemas import UserCreate
from typing import Optional, List
import uuid
from datetime import datetime

# Clase UserRepository
class UserRepository:
    def __init__(self, db: Session):
        self.db = db
    
    # Obtener usuario por username
    def get_by_username(self, username: str) -> Optional[User]:
        return self.db.query(User).filter(User.username == username).first()
    
    # Obtener usuario por email
    def get_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()
    
    # Obtener usuario por id
    def get_by_id(self, id: str) -> Optional[User]:
        return self.db.query(User).filter(User.id == id).first()
    
    # Obtener usuario por token
    def get_by_token(self, token: str) -> Optional[User]:
        return self.db.query(User).filter(User.token == token).first()
    
    # Crear un usuario
    def create(self, user_data: dict) -> User:
        db_user = User(**user_data)
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user
    
    # Actualizar un usuario
    def update(self, user: User, update_data: dict) -> User:
        for key, value in update_data.items():
            setattr(user, key, value)
        user.updatedAt = datetime.utcnow()
        self.db.commit()
        self.db.refresh(user)
        return user
    
    # Guardar token
    def save_token(self, user: User, token: str, expire_at: datetime) -> User:
        user.token = token
        user.expireAt = expire_at
        self.db.commit()
        self.db.refresh(user)
        return user
    
    # Contar usuarios
    def count(self) -> int:
         return self.db.query(func.count(User.id)).scalar()
    
    # Eliminar todos los usuarios
    def delete_all(self) -> None:
        self.db.query(User).delete()
        self.db.commit()
