import pytest
from unittest.mock import patch, MagicMock
from httpx import Response, ConnectError
from fastapi import HTTPException
from app.services.orchestrator import OrchestratorService
from app.schemas import OfferCreateRequest

@pytest.mark.asyncio
async def test_get_current_user_success():
    service = OrchestratorService()
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = Response(200, json={"id": "user1"})
        res = await service.get_current_user("token")
        assert res["id"] == "user1"

@pytest.mark.asyncio
async def test_get_current_user_401():
    service = OrchestratorService()
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = Response(401)
        try:
            await service.get_current_user("token")
        except HTTPException as e:
            assert e.status_code == 401

@pytest.mark.asyncio
async def test_get_current_user_timeout():
    service = OrchestratorService()
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = ConnectError("Timeout")
        try:
            await service.get_current_user("token")
        except HTTPException as e:
            assert e.status_code == 503

@pytest.mark.asyncio
async def test_get_post_success():
    service = OrchestratorService()
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = Response(200, json={"id": "post1"})
        res = await service.get_post("post1")
        assert res["id"] == "post1"

@pytest.mark.asyncio
async def test_get_post_404():
    service = OrchestratorService()
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = Response(404)
        try:
            await service.get_post("post1")
        except HTTPException as e:
            assert e.status_code == 404

@pytest.mark.asyncio
async def test_get_post_timeout():
    service = OrchestratorService()
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = ConnectError("Timeout")
        try:
            await service.get_post("post1")
        except HTTPException as e:
            assert e.status_code == 503

@pytest.mark.asyncio
async def test_get_route_success():
    service = OrchestratorService()
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = Response(200, json={"id": "route1"})
        res = await service.get_route("route1")
        assert res["id"] == "route1"

@pytest.mark.asyncio
async def test_get_route_timeout():
    service = OrchestratorService()
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = ConnectError("Timeout")
        try:
            await service.get_route("route1")
        except HTTPException as e:
            assert e.status_code == 503

@pytest.mark.asyncio
async def test_create_offer_success():
    service = OrchestratorService()
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = Response(201, json={"id": "offer1"})
        req = OfferCreateRequest(description="A", size="LARGE", fragile=False, offer=100)
        res = await service.create_offer("post1", "user1", req)
        assert res["id"] == "offer1"

@pytest.mark.asyncio
async def test_create_offer_timeout():
    service = OrchestratorService()
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = ConnectError("Timeout")
        req = OfferCreateRequest(description="A", size="LARGE", fragile=False, offer=100)
        try:
            await service.create_offer("post1", "user1", req)
        except HTTPException as e:
            assert e.status_code == 503

@pytest.mark.asyncio
async def test_delete_offer_timeout():
    service = OrchestratorService()
    with patch("httpx.AsyncClient.delete") as mock_del:
        mock_del.side_effect = ConnectError("Timeout")
        await service.delete_offer("offer1") # Should pass silently

@pytest.mark.asyncio
async def test_save_score_success():
    service = OrchestratorService()
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = Response(201, json={"id": "score1"})
        res = await service.save_score("offer1", 100)
        assert res["id"] == "score1"

@pytest.mark.asyncio
async def test_save_score_timeout():
    service = OrchestratorService()
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = ConnectError("Timeout")
        try:
            await service.save_score("offer1", 100)
        except HTTPException as e:
            assert e.status_code == 503
