from sqlalchemy.orm import Session
from app.models import Score
from app.schemas import ScoreCreate

class ScoreService:
    def __init__(self, db: Session):
        self.db = db

    def create(self, score_data: ScoreCreate):
        new_score = Score(
            offerId=str(score_data.offerId),
            score=score_data.score
        )
        self.db.add(new_score)
        self.db.commit()
        self.db.refresh(new_score)
        return new_score

    def get_by_offer(self, offer_id: str):
        return self.db.query(Score).filter(Score.offerId == offer_id).first()

    def count(self):
        return self.db.query(Score).count()

    def reset(self):
        self.db.query(Score).delete()
        self.db.commit()
