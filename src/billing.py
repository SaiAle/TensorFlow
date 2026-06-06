from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel


class PriceIn(BaseModel):
    stripe_price_id: str
    amount_cents: int
    currency: str = "usd"


class SubscriptionOut(BaseModel):
    id: str
    plan: str
    status: str


def create_app(data_dir: Path) -> FastAPI:
    app = FastAPI(title="Billing stub")
    subs_file = data_dir / "subscriptions.json"

    @app.get("/")
    def health() -> dict[str, str]:
        return {"service": "billing", "status": "stub"}

    @app.get("/plans")
    def list_plans() -> list[dict]:
        return [
            {"id": "free", "amount_cents": 0, "currency": "usd"},
            {"id": "pro", "amount_cents": 999, "currency": "usd"},
        ]

    @app.post("/subscriptions/webhook")
    async def webhook() -> dict[str, str]:
        return {"accepted": "true"}

    @app.get("/subscriptions/{customer_id}", response_model=SubscriptionOut)
    def get_subscription(customer_id: str) -> SubscriptionOut:
        return SubscriptionOut(id=customer_id, plan="free", status="active")

    return app
