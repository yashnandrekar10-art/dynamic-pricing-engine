from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from optimizer import DynamicPricingEngine
app = FastAPI(title="Dynamic Pricing Optimization API",version="1.0.0")
engine=DynamicPricingEngine("demand_model.joblib")
class PricingRequest(BaseModel):
    product_id: str
    unit_price: float = Field(..., gt=0, description="Current selling price")
    unit_cost: float = Field(..., gt=0, description="Cost of goods sold / wholesale cost")
    stock_available: int = Field(..., gt=0, description="Current warehouse inventory")
    comp_1: float = Field(..., gt=0, description="Primary competitor price")
    comp_2: float = Field(..., gt=0, description="Secondary competitor price")
    freight_price: float = Field(..., ge=0, description="Shipping cost")
    product_score: float = Field(..., ge=1.0, le=5.0, description="Review score (1-5)")
    customers: int = Field(..., ge=0, description="Customer search volume / footfall")
    month: int = Field(..., ge=1, le=12, description="Month of year (1-12)")
    holiday: int = Field(0, ge=0, le=1, description="1 if holiday, 0 otherwise")
    weekday: int = Field(1, ge=0, le=1, description="1 if weekday, 0 otherwise")
    weekend: int = Field(0, ge=0, le=1, description="1 if weekend, 0 otherwise")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend-price")
def recommend_price(payload: PricingRequest):
    try:
        data = payload.model_dump()
        product_id = data.pop("product_id")
        unit_cost = data.pop("unit_cost")
        stock = data.pop("stock_available")

        result = engine.optimize_price(
            context=data,
            unit_cost=unit_cost,
            stock_available=stock
        )
        return {"product_id": product_id, "recommendation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {
        "message": "Dynamic Pricing API is live!",
        "docs_url": "http://127.0.0.1:8000/docs"
    }

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app,host="127.0.0.1", port=8000)