import os
os.environ["TMPDIR"] = "C:\\temp"
os.environ["TEMP"] = "C:\\temp"
os.environ["TMP"] = "C:\\temp"

from dotenv import load_model, load_dotenv
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

class AnalysisRequest(BaseModel):
    salesList: List[SalesData]

@app.post("/api/v1/kiosk/sales-analysis")
def analyze_kiosk_sales(payload: AnalysisRequest):
    sales_text = "\n".join([
        f"- {item.menuName} ({item.category}): {item.count}개 판매, 매출 {int(item.price * item.count):,}원"
        for item in payload.salesList
    ])
    
    prompt = f"""당신은 카페 매출 데이터 분석 전문가입니다. 다음 매출 데이터를 분석해주세요:

{sales_text}

다음 항목을 포함해서 분석해주세요:
1. 현재 매출 트렌드 요약
2. 인기 메뉴와 부진 메뉴 분석
3. 향후 매출 개선을 위한 구체적인 제안 2-3가지

친근하고 실용적인 톤으로 작성해주세요."""

    response = model_gemini.generate_content(prompt)
    
    return {
        "status": "success",
        "analysis": response.text
    }

# Prophet AI
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

    insight_prompt = f"""당신은 카페 매출 데이터 분석가입니다. 다음 Prophet 시계열 예측 결과를 보고 사장님께 전달할 인사이트를 작성하세요.

    최근 6개월 실측 매출 추이: {[round(v) for v in df['y'].tolist()]}
    다음 3개월 예측 매출: {[round(forecast['yhat'].iloc[len(df)+i]) for i in range(3)]}
    예상 성장률: {growth_rate}%

    다음 형식으로 2~3문장 작성하세요:
    1. 최근 추세에 대한 구체적 평가 (단순 "성장 중" 같은 뻔한 말 금지, 구체적 수치나 패턴 언급)
    2. 다음 달 예측에 대한 실질적 코멘트
    3. 이 추세를 활용하기 위한 실행 가능한 제안 1가지

    친근하지만 전문적인 톤으로, 사장님이 바로 행동할 수 있게 구체적으로 작성하세요."""

    insight_response = model_gemini.generate_content(insight_prompt)

    return {
        "forecastData": result,
        "growthRate": growth_rate,
        "insight": insight_response.text.strip()
    }