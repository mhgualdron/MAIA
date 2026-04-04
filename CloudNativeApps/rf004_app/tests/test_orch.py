import pytest
from httpx import AsyncClient, Response, ASGITransport
from app.main import app
from unittest.mock import patch
from uuid import uuid4

@pytest.mark.asyncio
async def test_create_offer_orchestrated_success():
    post_id = str(uuid4())
    route_id = str(uuid4())
    user_id = str(uuid4())
    token = "fake-token"
    payload = {
        "description": "Test Offer",
        "size": "MEDIUM",
        "fragile": True,
        "offer": 100.0
    }
    
    with patch("app.services.orchestrator.OrchestratorService.get_current_user") as mock_user, \
         patch("app.services.orchestrator.OrchestratorService.get_post") as mock_post_req, \
         patch("app.services.orchestrator.OrchestratorService.get_route") as mock_route, \
         patch("app.services.orchestrator.OrchestratorService.create_offer") as mock_create, \
         patch("app.services.orchestrator.OrchestratorService.save_score") as mock_save, \
         patch("httpx.AsyncClient.get") as mock_http_get: # Only for ping
         
            mock_user.return_value = {"id": user_id, "username": "testuser"}
            mock_post_req.return_value = {"id": post_id, "expireAt": "2099-01-01T00:00:00Z", "userId": str(uuid4()), "routeId": route_id}
            mock_route.return_value = {"id": route_id, "bagCost": 20}
            mock_create.return_value = {"id": str(uuid4()), "postId": post_id, "userId": user_id, "createdAt": "2023-11-01T10:00:00Z"}
            mock_save.return_value = {"id": str(uuid4())}
            
            mock_http_get.return_value = Response(200)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    f"/rf004/posts/{post_id}/offers", 
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"}
                )
            
            assert response.status_code == 201
            data = response.json()
            assert "data" in data
            assert data["data"]["postId"] == post_id
            assert data["msg"] == "Oferta creada exitosamente con utilidad calculada"

@pytest.mark.asyncio
async def test_create_offer_consistency_failure():
    # Test that if score creation fails, we try to delete the offer and return 503
    post_id = str(uuid4())
    route_id = str(uuid4())
    user_id = str(uuid4())
    offer_id = str(uuid4())
    token = "fake-token"
    payload = {"description": "X", "size": "SMALL", "fragile": False, "offer": 50.0}
    
    with patch("app.services.orchestrator.OrchestratorService.get_current_user") as mock_user, \
         patch("app.services.orchestrator.OrchestratorService.get_post") as mock_post_req, \
         patch("app.services.orchestrator.OrchestratorService.get_route") as mock_route, \
         patch("app.services.orchestrator.OrchestratorService.create_offer") as mock_create, \
         patch("app.services.orchestrator.OrchestratorService.save_score") as mock_save, \
         patch("app.services.orchestrator.OrchestratorService.delete_offer") as mock_delete, \
         patch("httpx.AsyncClient.get") as mock_http_get:
        
        mock_user.return_value = {"id": user_id, "username": "testuser"}
        mock_post_req.return_value = {"id": post_id, "expireAt": "2099-01-01T00:00:00Z", "userId": str(uuid4()), "routeId": route_id}
        mock_route.return_value = {"id": route_id, "bagCost": 10}
        mock_create.return_value = {"id": offer_id, "postId": post_id, "userId": user_id, "createdAt": "X"}
        mock_http_get.return_value = Response(200)

        # Force failure in save_score
        mock_save.side_effect = Exception("Failure")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                f"/rf004/posts/{post_id}/offers", 
                json=payload,
                headers={"Authorization": f"Bearer {token}"}
            )
            
        assert response.status_code == 503
        mock_delete.assert_called_once()
