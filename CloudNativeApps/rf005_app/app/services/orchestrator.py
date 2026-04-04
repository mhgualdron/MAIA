import httpx
import asyncio
from fastapi import HTTPException
from app.config import settings

class OrchestratorService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=3.0)

    async def close(self):
        await self.client.aclose()

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
        url = f"{settings.users_service_url}/users/me"
        headers = {"Authorization": f"Bearer {token}"}
        response = await self.safe_call("GET", url, headers=headers)
        if response.status_code == 200:
            return response.json()
            
        # Re-verify service health if we get an auth error
        if response.status_code in (401, 403):
            try:
                await self.ping_service("USERS_RECHECK", f"{settings.users_service_url}/users/ping")
                raise HTTPException(status_code=401, detail="El token no es válido o está vencido.")
            except HTTPException as e:
                raise e
            except:
                raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")
        
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def get_post(self, post_id: str):
        url = f"{settings.posts_service_url}/posts/{post_id}"
        response = await self.safe_call("GET", url)
        if response.status_code == 200:
            return response.json()
            
        # Re-verify service health if we get a 404
        if response.status_code == 404:
            try:
                await self.ping_service("POSTS_RECHECK", f"{settings.posts_service_url}/posts/ping")
                raise HTTPException(status_code=404, detail="La publicación no existe.")
            except HTTPException as e:
                raise e
            except:
                raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")
        
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def get_route(self, route_id: str):
        url = f"{settings.routes_service_url}/routes/{route_id}"
        response = await self.safe_call("GET", url)
        if response.status_code == 200:
            return response.json()
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def get_offers(self, post_id: str):
        url = f"{settings.offers_service_url}/offers"
        params = {"post": post_id}
        response = await self.safe_call("GET", url, params=params)
        if response.status_code == 200:
            return response.json()
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def get_score(self, offer_id: str):
        """Fetches the score. Returns None if it fails or doesn't exist to ensure resilience."""
        url = f"{settings.scores_service_url}/scores/{offer_id}"
        try:
            response = await self.client.get(url, timeout=1.5)
            if response.status_code == 200:
                data = response.json()
                return data.get("score")
            return None
        except Exception:
            return None

    async def process_rf005(self, post_id: str, token: str):
        # Resilience check: fast liveness ping for all dependencies
        await self.ping_service("USERS", f"{settings.users_service_url}/users/ping")
        await self.ping_service("POSTS", f"{settings.posts_service_url}/posts/ping")
        await self.ping_service("OFFERS", f"{settings.offers_service_url}/ping")
        await self.ping_service("ROUTES", f"{settings.routes_service_url}/routes/ping")

        user = await self.get_current_user(token)
        post = await self.get_post(post_id)

        # Ensure only the owner can access this information
        if post.get("userId") != user.get("id"):
            raise HTTPException(status_code=403, detail="El usuario no tiene permiso para ver el contenido de esta publicación.")

        # Fetch route and offers concurrently since they don't depend on each other at this stage
        route_task = asyncio.create_task(self.get_route(post["routeId"]))
        offers_task = asyncio.create_task(self.get_offers(post_id))
        
        route, base_offers = await asyncio.gather(route_task, offers_task)

        # Enhance offers with scores concurrently
        score_tasks = [self.get_score(offer["id"]) for offer in base_offers]
        scores = await asyncio.gather(*score_tasks)

        enriched_offers = []
        for index, offer in enumerate(base_offers):
            offer["score"] = scores[index]
            enriched_offers.append(offer)

        # Sort the offers descending by score, treating None as the lowest possible score (-infinity)
        enriched_offers.sort(key=lambda x: x["score"] if x["score"] is not None else float('-inf'), reverse=True)

        return {
            "data": {
                "id": post["id"],
                "expireAt": post["expireAt"],
                "route": {
                    "id": route["id"],
                    "flightId": route["flightId"],
                    "origin": {
                        "airportCode": route["sourceAirportCode"],
                        "country": route["sourceCountry"]
                    },
                    "destiny": {
                        "airportCode": route["destinyAirportCode"],
                        "country": route["destinyCountry"]
                    },
                    "bagCost": route["bagCost"]
                },
                "plannedStartDate": route["plannedStartDate"],
                "plannedEndDate": route["plannedEndDate"],
                "createdAt": post["createdAt"],
                "offers": enriched_offers
            }
        }
