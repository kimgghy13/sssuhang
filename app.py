
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. 화면 설정
st.set_page_config(
    page_title="태양광 발전량 예측",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ 태양광 발전량 예측 시스템")
st.write("선형회귀 머신러닝을 이용한 태양광 발전량 예측")

# 2. 예시 데이터 자동 생성
@st.cache_data
def make_data():
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        "기온": np.random.uniform(5, 38, n),
        "습도": np.random.uniform(20, 95, n),
        "풍속": np.random.uniform(0, 12, n),
        "일사량": np.random.uniform(0.5, 6, n),
        "지역": np.random.choice(
            ["서울", "경기", "강원", "충청", "전라", "경상"],
            n
        ),
        "태양광 설비용량": np.random.uniform(1, 100, n)
    })

    # 예시 발전량 계산
    weather_factor = (
        1 - abs(df["기온"] - 25) * 0.003
    )

    df["태양광 발전량 예측값 총합(MWh)"] = (
        df["일사량"]
        * df["태양광 설비용량"]
        * 0.18
        * weather_factor
        * (1 - df["습도"] * 0.001)
        + np.random.normal(0, 0.5, n)
    )

    df["태양광 발전량 예측값 총합(MWh)"] = (
        df["태양광 발전량 예측값 총합(MWh)"].clip(lower=0)
    )

    return df


df = make_data()

# 3. 데이터 분석
st.header("1. 데이터 분석")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(df)}개")
col2.metric("독립 변수", "6개")
col3.metric("종속 변수", "1개")

st.dataframe(df.head(10), use_container_width=True)

st.info(
    "현재는 실습용 예시 데이터 500개를 자동 생성하여 사용합니다. "
    "실제 발전소 측정 데이터는 아닙니다."
)

# 4. 독립 변수와 종속 변수 설정
st.header("2. 머신러닝 변수 설정")

features = [
    "기온",
    "습도",
    "풍속",
    "일사량",
    "지역",
    "태양광 설비용량"
]

target = "태양광 발전량 예측값 총합(MWh)"

X = df[features]
y = df[target]

st.write("독립 변수:", ", ".join(features))
st.write("종속 변수:", target)

# 5. 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. 선형회귀 모델 만들기
st.header("3. 선형회귀 모델 학습")

preprocessor = ColumnTransformer(
    transformers=[
        ("region", OneHotEncoder(handle_unknown="ignore"), ["지역"])
    ],
    remainder="passthrough"
)

model = make_pipeline(
    preprocessor,
    LinearRegression()
)

model.fit(X_train, y_train)

st.success("머신러닝 모델 학습이 완료되었습니다!")

# 7. 발전량 예측
st.header("4. 태양광 발전량 예측")

predictions = model.predict(X_test)

result = pd.DataFrame({
    "실제 발전량(MWh)": y_test.values,
    "예측 발전량(MWh)": predictions
})

result["오차(MWh)"] = (
    result["실제 발전량(MWh)"]
    - result["예측 발전량(MWh)"]
)

st.dataframe(result.head(20), use_container_width=True)

# 8. 정확도 평가
st.header("5. 모델 성능 평가")

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

c1, c2, c3 = st.columns(3)

c1.metric("R² 결정계수", f"{r2:.4f}")
c2.metric("MAE 평균 오차", f"{mae:.4f} MWh")
c3.metric("RMSE 평균 오차", f"{rmse:.4f} MWh")

st.caption("R²는 1에 가까울수록 좋고, MAE와 RMSE는 0에 가까울수록 좋습니다.")

# 9. 실제값과 예측값 비교 그래프
st.header("6. 실제 발전량과 예측 발전량 비교")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    range(len(y_test)),
    y_test.values,
    label="Actual",
    marker="o"
)

ax.plot(
    range(len(predictions)),
    predictions,
    label="Predicted",
    marker="x"
)

ax.set_xlabel("Test Data")
ax.set_ylabel("Power Generation (MWh)")
ax.set_title("Actual vs Predicted Solar Power")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# 10. 오차 그래프
st.header("7. 예측 오차 그래프")

st.bar_chart(result["오차(MWh)"])

st.success("태양광 발전량 예측 분석이 모두 완료되었습니다!")

# 11. 결과 다운로드
csv = result.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="예측 결과 CSV 다운로드",
    data=csv,
    file_name="solar_prediction_result.csv",
    mime="text/csv"
)
