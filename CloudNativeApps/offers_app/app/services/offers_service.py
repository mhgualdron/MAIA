from sqlalchemy.orm import Session
from app.models import Offer
from app.schemas import OfferCreate, OfferResponse, OfferBase
from uuid import UUID

class OfferService:
    def __init__(self, db: Session):
        self.db = db

    def create(self, offer: OfferCreate):
        new_offer = Offer(
            userId=str(offer.userId),
            postId=str(offer.postId),
            description=offer.description,
            size=offer.size,
            fragile=offer.fragile,
            offer=offer.offer,
        )
        self.db.add(new_offer)
        self.db.commit()
        self.db.refresh(new_offer)
        return new_offer

    def list_offers(self, postId: str = None, userId: str = None):
        query = self.db.query(Offer)
        if postId:
            query = query.filter(Offer.postId == str(postId))
        if userId:
            query = query.filter(Offer.userId == str(userId))
        return query.all()

    def get_offer(self, offer_id: str):
        return self.db.query(Offer).filter(Offer.id == offer_id).first()

    def delete_offer(self, offer_id: str):
        offer = self.get_offer(offer_id)
        if offer:
            self.db.delete(offer)
            self.db.commit()
            return True
        return False

    def count(self):
        return self.db.query(Offer).count()

    def reset(self):
        self.db.query(Offer).delete()
        self.db.commit()
