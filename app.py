
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# --------------------------------
# 1. 웹사이트 기본 설정
# --------------------------------

st.set_page_config(
    page_title="태양광 발전량 예측",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ 태양광 발전량 예측 시스템")
st.write("선형회귀 머신러닝을 이용한 발전량 예측")


# --------------------------------
# 2. CSV 파일 자동 불러오기
# --------------------------------

st.header("1. 데이터 불러오기")

file_name = "한국동서발전(주)_전국 태양광 발전량 예측값_20241203 (1).csv"

file_path = Path(__file__).parent / file_name

if not file_path.exists():
    st.error("CSV 파일을 찾을 수 없습니다.")
    st.write("GitHub 저장소의 app.py와 같은 위치에 CSV 파일을 올려 주세요.")
    st.write("필요한 파일 이름:", file_name)
    st.stop()

# 한글 및 CP949 인코딩 처리
df = None

for encoding in ["cp949", "utf-8-sig", "euc-kr", "utf-8"]:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        st.success("CSV 파일 자동 불러오기 성공!")
        break
    except UnicodeDecodeError:
        continue

if df is None:
    st.error("CSV 파일의 인코딩을 읽을 수 없습니다.")
    st.stop()

df.columns = df.columns.str.strip()

st.write("전체 데이터 크기:", df.shape)
st.dataframe(df.head())


# --------------------------------
# 3. 독립 변수와 종속 변수 설정
# --------------------------------

st.header("2. 데이터 전처리")

features = [
    "기온",
    "습도",
    "풍속",
    "일사량 예측값",
    "시도명",
    "전체 태양광 설비용량(MW)"
]

target = "태양광 발전량 예측값 총합(MWh)"

# 필요한 컬럼이 있는지 확인
missing = [
    col for col in features + [target]
    if col not in df.columns
]

if missing:
    st.error("다음 컬럼이 CSV에 없습니다.")
    st.write(missing)
    st.write("실제 CSV 컬럼명:", df.columns.tolist())
    st.stop()

data = df[features + [target]].copy()

# 숫자 데이터 변환
numeric_features = [
    "기온",
    "습도",
    "풍속",
    "일사량 예측값",
    "전체 태양광 설비용량(MW)"
]

for col in numeric_features + [target]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# 발전량이 없는 행 제거
data = data.dropna(subset=[target])

if len(data) < 5:
    st.error("학습할 데이터가 부족합니다. 최소 5개 이상의 행이 필요합니다.")
    st.stop()

X = data[features]
y = data[target]

st.write("분석에 사용한 데이터:", len(data), "개")


# --------------------------------
# 4. 데이터 전처리
# --------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, ["시도명"])
])


# --------------------------------
# 5. 학습 데이터와 테스트 데이터 분리
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------
# 6. 선형회귀 모델 학습
# --------------------------------

st.header("3. 머신러닝 모델 학습")

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regression", LinearRegression())
])

model.fit(X_train, y_train)

st.success("선형회귀 모델 학습 완료!")


# --------------------------------
# 7. 발전량 예측
# --------------------------------

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

st.dataframe(result.reset_index(drop=True))


# --------------------------------
# 8. 모델 성능 평가
# --------------------------------

st.header("5. 모델 정확도 평가")

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

col1, col2, col3 = st.columns(3)

col1.metric("R² 결정계수", f"{r2:.4f}")
col2.metric("MAE 평균 오차", f"{mae:.4f} MWh")
col3.metric("RMSE 평균 오차", f"{rmse:.4f} MWh")

st.info(
    "R²는 1에 가까울수록 좋고, "
    "MAE와 RMSE는 0에 가까울수록 좋습니다."
)


# --------------------------------
# 9. 실제 발전량과 예측 발전량 그래프
# --------------------------------

st.header("6. 실제 발전량과 예측 발전량 비교")

st.line_chart(
    result[["실제 발전량(MWh)", "예측 발전량(MWh)"]]
)

st.subheader("실제값과 예측값의 오차")

st.bar_chart(result["오차(MWh)"])

st.success("모든 분석과 예측이 완료되었습니다!")

# --------------------------------
# 10. 최종 데이터 정보
# --------------------------------

st.header("7. 최종 분석 결과")

st.write("전체 데이터:", len(data), "개")
st.write("학습 데이터:", len(X_train), "개")
st.write("테스트 데이터:", len(X_test), "개")
st.write("평균 절대 오차:", round(mae, 4), "MWh")
st.write("평균 제곱근 오차:", round(rmse, 4), "MWh")
st.write("결정계수:", round(r2, 4))
