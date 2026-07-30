
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from prophet import Prophet

import google.generativeai as genai
import pandas as pd

load_dotenv()
app = FastAPI()

gcp_api_key = os.getenv("GCP_API_KEY")

# Gemini AI
# CORS 설정 - React에서 직접 호출하니까 필요해요
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 나중에 배포 시 React 주소로 제한
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key = gcp_api_key)
model_gemini = genai.GenerativeModel("gemini-2.5-flash")

class SalesData(BaseModel):
    menuName: str
    category: str
    price: float
    count: int

class DailyStats(BaseModel):
    day: str
    orderCount: int
    revenue: float

class HourlyStats(BaseModel):
    hour: str
    orderCount: int
    revenue: float

class AnalysisRequest(BaseModel):
    salesList: List[SalesData]
    dailyStats: List[DailyStats] = []
    hourlyStats: List[HourlyStats] = []

# Gemini AI 텍스트 기반 예측
@app.post("/api/v1/kiosk/sales-analysis")
def analyze_kiosk_sales(payload: AnalysisRequest):
    sales_text = "\n".join([
        f"- {item.menuName} ({item.category}): {item.count}개 판매, 매출 {int(item.price * item.count):,}원"
        for item in payload.salesList
    ])

    daily_text = "\n".join([
        f"- {item.day}요일: 주문 {item.orderCount}건, 매출 {int(item.revenue):,}원"
        for item in payload.dailyStats
    ])

    hourly_text = "\n".join([
        f"- {item.hour}: 주문 {item.orderCount}건, 매출 {int(item.revenue):,}원"
        for item in payload.hourlyStats
    ])
    
    prompt = f"""You are an expert cafe business consultant analyzing real sales data.
    Based on the following data, provide sharp and actionable insights for the cafe owner.

    [Menu Sales Data]
    {sales_text}

    [Sales by Day of Week]
    {daily_text}

    [Sales by Hour]
    {hourly_text}


    Analyze the data and write in Korean with the following structure:
    1. Sales trend summary (best-selling menus, category trends with specific numbers)
    2. Day of week and hourly pattern analysis (when is it busiest and slowest, use exact figures)
    3. Two immediately actionable improvement suggestions (data-driven and specific, not generic)

    Rules:
    - Write the entire response in Korean
    - Always reference specific numbers from the data
    - Avoid vague or generic business advice
    - Use a friendly but professional tone suitable for a small cafe owner"""

    response = model_gemini.generate_content(prompt)
    
    return {
        "status": "success",
        "analysis": response.text
    }

# Prophet AI 그래프 기반 예측
class MonthlyRevenue(BaseModel):
    month: str  # "2026-01" 형태
    revenue: float

class ForecastRequest(BaseModel):
    monthlyData: List[MonthlyRevenue]

@app.post("/api/v1/kiosk/sales-forecast")
def forecast_sales(payload: ForecastRequest):
    # 1. Prophet이 요구하는 형식으로 변환 (ds: 날짜, y: 값)
    df = pd.DataFrame([
        {"ds": item.month + "-01", "y": item.revenue}
        for item in payload.monthlyData
    ])
    df["ds"] = pd.to_datetime(df["ds"])

    # 2. Prophet 모델 학습
    model = Prophet()
    model.fit(df)

    # 3. 향후 3개월 예측
    future = model.make_future_dataframe(periods=3, freq='MS')
    forecast = model.predict(future)

    # 4. 결과 정리 (실측 + 예측 구분)
    result = []
    for i, row in forecast.iterrows():
        is_predicted = i >= len(df)
        result.append({
            "month": row["ds"].strftime("%Y-%m"),
            "value": round(row["yhat"]),
            "isPredicted": is_predicted
        })

    # 5. Gemini한테 인사이트 요청
    last_actual = df["y"].iloc[-1]
    next_predicted = forecast["yhat"].iloc[len(df)]
    growth_rate = round((next_predicted - last_actual) / last_actual * 100, 1)

    insight_prompt = f"""You are an expert cafe sales analyst. 
    Analyze the following Prophet time-series forecast results and provide insights for the store owner.

    Recent 6 months actual revenue: {[round(v) for v in df['y'].tolist()]}
    Next 3 months predicted revenue: {[round(forecast['yhat'].iloc[len(df)+i]) for i in range(3)]}
    Expected growth rate: {growth_rate}%

    Write 2-3 sentences in Korean following this format:
    1. A specific evaluation of the recent trend (avoid vague phrases like "growing steadily", mention specific numbers or patterns)
    2. A practical comment on next month's forecast
    3. One actionable suggestion the owner can implement immediately

    Rules:
    - Write the entire response in Korean
    - Always reference specific numbers from the data
    - Avoid vague or generic business advice
    - Use a friendly but professional tone suitable for a small cafe owner"""

    insight_response = model_gemini.generate_content(insight_prompt)

    return {
        "forecastData": result,
        "growthRate": growth_rate,
        "insight": insight_response.text.strip()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)