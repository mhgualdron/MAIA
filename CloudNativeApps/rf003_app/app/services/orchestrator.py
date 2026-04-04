import os
import httpx
from fastapi import HTTPException, status
from datetime import datetime, timezone
from app.schemas import PostCreateRequest

USERS_SERVICE_URL = os.getenv("USERS_SERVICE_URL", "http://users-app-service")
POSTS_SERVICE_URL = os.getenv("POSTS_SERVICE_URL", "http://posts-app-service")
ROUTES_SERVICE_URL = os.getenv("ROUTES_SERVICE_URL", "http://routes-app-service")

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
                # If ping succeeded, it's a real business error from a healthy service
                raise HTTPException(status_code=401, detail="El token no es válido o está vencido.")
            except HTTPException as e:
                # Propagate our own 401 or the 503 from ping
                raise e
            except:
                raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")
        
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def get_route_by_flight(self, flight_id: str):
        url = f"{ROUTES_SERVICE_URL}/routes"
        params = {"flight": flight_id}
        response = await self.safe_call("GET", url, params=params)
        if response.status_code == 200:
            return response.json()
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def create_route(self, request: PostCreateRequest):
        url = f"{ROUTES_SERVICE_URL}/routes"
        payload = {
            "flightId": str(request.flightId),
            "sourceAirportCode": request.origin.airportCode,
            "sourceCountry": request.origin.country,
            "destinyAirportCode": request.destiny.airportCode,
            "destinyCountry": request.destiny.country,
            "bagCost": request.bagCost,
            "plannedStartDate": request.plannedStartDate,
            "plannedEndDate": request.plannedEndDate
        }
        
        response = await self.safe_call("POST", url, json=payload)
        if response.status_code == 201:
            return response.json()
        if response.status_code == 412:
            detail = response.json().get("msg", "Error creating route")
            raise HTTPException(status_code=412, detail=detail)
        # Any other error is 503
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def delete_route(self, route_id: str):
        url = f"{ROUTES_SERVICE_URL}/routes/{route_id}"
        try:
            await self.safe_call("DELETE", url)
        except Exception:
            pass

    async def get_posts_by_route_and_owner(self, route_id: str, user_id: str):
        url = f"{POSTS_SERVICE_URL}/posts"
        params = {"route": route_id, "owner": user_id}
        response = await self.safe_call("GET", url, params=params)
        if response.status_code == 200:
            return response.json()
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def create_post(self, route_id: str, user_id: str, expire_at: str):
        url = f"{POSTS_SERVICE_URL}/posts"
        payload = {
            "routeId": str(route_id),
            "userId": str(user_id),
            "expireAt": expire_at
        }
        
        response = await self.safe_call("POST", url, json=payload)
        if response.status_code == 201:
            return response.json()
        if response.status_code == 412:
            detail = response.json().get("msg", "La fecha de expiración no es válida")
            raise HTTPException(status_code=412, detail=detail)
        raise HTTPException(status_code=503, detail="El servicio está temporalmente fuera de servicio.")

    async def process_rf003(self, token: str, request: PostCreateRequest):
        # Resilience check: fast liveness ping for all dependencies
        await self.ping_service("USERS", f"{USERS_SERVICE_URL}/users/ping")
        await self.ping_service("POSTS", f"{POSTS_SERVICE_URL}/posts/ping")
        await self.ping_service("ROUTES", f"{ROUTES_SERVICE_URL}/routes/ping")

        user = await self.get_current_user(token)
        user_id = user["id"]

        routes_list = await self.get_route_by_flight(request.flightId)
        
        # Robust filtering: ensure we only use the route if it matches the flightId
        target_route = next((r for r in routes_list if str(r.get("flightId")) == str(request.flightId)), None) if isinstance(routes_list, list) else None
        
        route_created_by_us = False
        if target_route:
            route = target_route
            planned_start = route.get("plannedStartDate")
        else:
            planned_start = request.plannedStartDate
            route = await self.create_route(request)
            route_created_by_us = True

        route_id = route["id"]
        
        now_utc = datetime.now(timezone.utc)
        expire_dt = datetime.fromisoformat(request.expireAt.replace("Z", "+00:00"))
        start_dt = datetime.fromisoformat(planned_start.replace("Z", "+00:00"))
        
        if start_dt <= now_utc:
            if route_created_by_us: await self.delete_route(route_id)
            raise HTTPException(status_code=412, detail="Las fechas del trayecto no son válidas")
        
        if expire_dt <= now_utc or expire_dt > start_dt:
            if route_created_by_us: await self.delete_route(route_id)
            raise HTTPException(status_code=412, detail="La fecha expiración no es válida")

        try:
            existing_posts = await self.get_posts_by_route_and_owner(route_id, user_id)
        except HTTPException as e:
            if route_created_by_us: await self.delete_route(route_id)
            raise e
            
        if existing_posts:
            if route_created_by_us: await self.delete_route(route_id)
            raise HTTPException(status_code=412, detail="El usuario ya tiene una publicación para la misma fecha")

        try:
            post = await self.create_post(route_id, user_id, request.expireAt)
        except HTTPException as e:
            if route_created_by_us: await self.delete_route(route_id)
            raise e

        return {
            "data": {
                "id": post["id"],
                "userId": post["userId"],
                "createdAt": post["createdAt"],
                "expireAt": post.get("expireAt", request.expireAt),
                "route": {
                    "id": route_id,
                    "createdAt": route.get("createdAt", post.get("createdAt"))
                }
            },
            "msg": "Publicación creada con éxito"
        }
