import os
import httpx
from fastapi import HTTPException, status
from datetime import datetime, timezone
from app.schemas import OfferCreateRequest

USERS_SERVICE_URL = os.getenv("USERS_SERVICE_URL", "http://users-app-service")
POSTS_SERVICE_URL = os.getenv("POSTS_SERVICE_URL", "http://posts-app-service")
OFFERS_SERVICE_URL = os.getenv("OFFERS_SERVICE_URL", "http://offers-app-service")
ROUTES_SERVICE_URL = os.getenv("ROUTES_SERVICE_URL", "http://routes-app-service")
SCORES_SERVICE_URL = os.getenv("SCORES_SERVICE_URL", "http://scores-app-service")

class OrchestratorService:
    async def safe_call(self, method: str, url: str, **kwargs):
        timeout = kwargs.pop("timeout", 2.0)
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                response = await client.request(method, url, **kwargs)
                return response
        except Exception as e:
            raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def ping_service(self, name: str, url: str):
        """Quick liveness check - raises 503 immediately if service is unreachable or not 200."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(1.5), trust_env=False) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def get_current_user(self, token: str):
        url = f"{USERS_SERVICE_URL}/users/me"
        headers = {"Authorization": f"Bearer {token}"}
        response = await self.safe_call("GET", url, headers=headers)
        if response.status_code == 200:
            return response.json()
            
        # Re-verify service health if we get an auth error
        if response.status_code in (401, 403):
            try:
                await self.ping_service("USERS_RECHECK", f"{USERS_SERVICE_URL}/users/ping")
                raise HTTPException(status_code=401, detail="El token no es válido o está vencido.")
            except HTTPException as e:
                raise e
            except:
                raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")
        
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def get_post(self, post_id: str):
        url = f"{POSTS_SERVICE_URL}/posts/{post_id}"
        response = await self.safe_call("GET", url)
        if response.status_code == 200:
            return response.json()
            
        # Re-verify service health if we get a 404
        if response.status_code == 404:
            try:
                await self.ping_service("POSTS_RECHECK", f"{POSTS_SERVICE_URL}/posts/ping")
                raise HTTPException(status_code=404, detail="La publicación no existe.")
            except HTTPException as e:
                raise e
            except:
                raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")
        
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def get_route(self, route_id: str):
        url = f"{ROUTES_SERVICE_URL}/routes/{route_id}"
        response = await self.safe_call("GET", url)
        if response.status_code == 200:
            return response.json()
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    def calculate_score(self, offer_amount: float, size: str, bag_cost: float):
        occupancy = {"LARGE": 1.0, "MEDIUM": 0.5, "SMALL": 0.25}
        pct = occupancy.get(size, 0)
        return offer_amount - (pct * bag_cost)

    async def create_offer(self, post_id: str, user_id: str, request: OfferCreateRequest):
        url = f"{OFFERS_SERVICE_URL}/offers"
        payload = {
            "postId": post_id,
            "userId": user_id,
            "description": request.description,
            "size": request.size,
            "fragile": request.fragile,
            "offer": int(request.offer)
        }
        response = await self.safe_call("POST", url, json=payload)
        if response.status_code == 201:
            return response.json()
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def delete_offer(self, offer_id: str):
        try:
            await self.safe_call("DELETE", f"{OFFERS_SERVICE_URL}/offers/{offer_id}")
        except Exception:
            pass

    async def save_score(self, offer_id: str, calculated_score: float):
        url = f"{SCORES_SERVICE_URL}/scores"
        payload = {
            "offerId": offer_id,
            "score": calculated_score
        }
        response = await self.safe_call("POST", url, json=payload)
        if response.status_code == 201:
            return response.json()
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def process_rf004(self, post_id: str, token: str, request: OfferCreateRequest):
        # Resilience check: fast liveness ping for all dependencies
        await self.ping_service("USERS", f"{USERS_SERVICE_URL}/users/ping")
        await self.ping_service("POSTS", f"{POSTS_SERVICE_URL}/posts/ping")
        await self.ping_service("OFFERS", f"{OFFERS_SERVICE_URL}/ping")
        await self.ping_service("ROUTES", f"{ROUTES_SERVICE_URL}/routes/ping")

        # 1. Users Check
        user = await self.get_current_user(token)
        user_id = user["id"]

        # 2. Posts Check
        post = await self.get_post(post_id)
        
        # Business Rules
        if post["userId"] == user_id:
            raise HTTPException(status_code=412, detail="La publicación es del mismo usuario")
        
        expire_at = datetime.fromisoformat(post["expireAt"].replace('Z', '+00:00'))
        if expire_at < datetime.now(timezone.utc):
            raise HTTPException(status_code=412, detail="La publicación ya está expirada")

        # 4. Routes Check
        route = await self.get_route(post["routeId"])
        bag_cost = route["bagCost"]

        # 5. Create Offer
        offer = await self.create_offer(post_id, user_id, request)

        # 6. Save Score (Shielded with Rollback)
        try:
            calculated_score = self.calculate_score(request.offer, request.size, bag_cost)
            await self.save_score(offer["id"], calculated_score)
        except Exception as e:
            # Try to cleanup the offer if scoring fails
            try:
                await self.delete_offer(offer["id"])
            except:
                pass
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

        return {
            "data": {
                "id": offer["id"],
                "userId": offer["userId"],
                "createdAt": offer["createdAt"],
                "postId": offer["postId"]
            },
            "msg": "Oferta creada exitosamente con utilidad calculada"
        }
