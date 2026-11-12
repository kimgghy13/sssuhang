
# 필요한 라이브러리 불러오기
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# 1. 웹사이트 제목
st.title("☀️ 태양광 발전량 예측 시스템")
st.write("머신러닝 선형회귀 모델을 이용한 태양광 발전량 예측")


# 2. CSV 파일 불러오기
st.header("1. 데이터 불러오기")

file_name = "한국동서발전(주)_전국 태양광 발전량 예측값_20241203 (1).csv"

uploaded_file = st.file_uploader(
    "CSV 파일을 업로드하세요",
    type=["csv"]
)

# 깃허브에 CSV 파일이 있으면 자동으로 불러오기
if uploaded_file is not None:
    csv_data = uploaded_file.getvalue()
elif os.path.exists(file_name):
    with open(file_name, "rb") as f:
        csv_data = f.read()
else:
    csv_data = None

# 한글 CSV 파일의 인코딩 처리
if csv_data is not None:

    df = None

    for encoding in ["cp949", "utf-8-sig", "euc-kr"]:
        try:
            df = pd.read_csv(
                BytesIO(csv_data),
                encoding=encoding
            )
            st.success(f"파일 불러오기 성공! 인코딩: {encoding}")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        st.error("CSV 파일의 인코딩을 확인해 주세요.")
        st.stop()

    # 컬럼명 앞뒤 공백 제거
    df.columns = df.columns.str.strip()

    # 3. 데이터 확인
    st.header("2. 데이터 분석")

    st.write("데이터의 처음 5개 행")
    st.dataframe(df.head())

    st.write("데이터 크기")
    st.write(f"행: {df.shape[0]}개 / 열: {df.shape[1]}개")

    st.write("컬럼명")
    st.write(df.columns.tolist())

    # 4. 사용할 독립 변수와 종속 변수 설정
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
    missing_columns = [
        col for col in features + [target]
        if col not in df.columns
    ]

    if missing_columns:
        st.error(f"CSV 파일에 다음 컬럼이 없습니다: {missing_columns}")
        st.stop()

    # 필요한 데이터만 선택
    data = df[features + [target]].copy()

    # 발전량이 숫자로 변환되도록 처리
    data[target] = pd.to_numeric(
        data[target], errors="coerce"
    )

    # 발전량이 없는 행 제거
    data = data.dropna(subset=[target])

    # 5. 데이터 전처리
    numeric_features = [
        "기온",
        "습도",
        "풍속",
        "일사량 예측값",
        "전체 태양광 설비용량(MW)"
    ]

    categorical_features = ["시도명"]

    # 숫자 데이터 변환
    for col in numeric_features:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    # 숫자 데이터의 결측값은 중앙값으로 채우기
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # 지역 데이터의 결측값 처리 후 숫자로 변환
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 숫자 데이터와 지역 데이터를 각각 전처리
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    X = data[features]
    y = data[target]

    # 데이터가 충분한지 확인
    if len(data) < 5:
        st.error("머신러닝 학습을 위해 최소 5개 이상의 데이터가 필요합니다.")
        st.stop()

    # 6. 학습 데이터와 테스트 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7. 선형회귀 모델 만들기
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ])

    # 모델 학습
    model.fit(X_train, y_train)

    # 8. 발전량 예측
    predictions = model.predict(X_test)

    # 9. 모델 정확도 평가
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    st.header("3. 머신러닝 예측 결과")

    col1, col2, col3 = st.columns(3)

    col1.metric("R² 결정계수", f"{r2:.4f}")
    col2.metric("평균 절대 오차", f"{mae:.4f} MWh")
    col3.metric("평균 제곱근 오차", f"{rmse:.4f} MWh")

    st.info(
        "R²는 1에 가까울수록 좋습니다. "
        "MAE와 RMSE는 작을수록 예측 오차가 적습니다."
    )

    # 10. 실제값과 예측값 비교
    st.header("4. 실제 발전량과 예측 발전량 비교")

    result = pd.DataFrame({
        "실제 발전량(MWh)": y_test.values,
        "예측 발전량(MWh)": predictions
    })

    st.dataframe(result.reset_index(drop=True))

    # 11. 그래프 그리기
    st.header("5. 예측 결과 그래프")

    fig, ax = plt.subplots(figsize=(10, 5))

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

    ax.set_title("Actual vs Predicted Solar Power")
    ax.set_xlabel("Test Data")
    ax.set_ylabel("Power Generation (MWh)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # 12. 예측 오차 그래프
    st.subheader("실제값과 예측값의 차이")

    errors = y_test.values - predictions

    fig2, ax2 = plt.subplots(figsize=(10, 4))

    ax2.bar(range(len(errors)), errors)

    ax2.axhline(y=0, linestyle="--")
    ax2.set_title("Prediction Errors")
    ax2.set_xlabel("Test Data")
    ax2.set_ylabel("Error (MWh)")
    ax2.grid(True)

    st.pyplot(fig2)

    # 13. 최종 결과
    st.header("6. 최종 분석 결과")

    st.write(f"전체 데이터 수: {len(data)}개")
    st.write(f"학습 데이터 수: {len(X_train)}개")
    st.write(f"테스트 데이터 수: {len(X_test)}개")

    st.write(f"평균 절대 오차(MAE): {mae:.4f} MWh")
    st.write(f"평균 제곱근 오차(RMSE): {rmse:.4f} MWh")
    st.write(f"결정계수(R²): {r2:.4f}")

    st.success("태양광 발전량 예측이 완료되었습니다!")

else:
    st.warning("CSV 파일을 업로드하거나 깃허브 저장소에 CSV 파일을 넣어주세요.")
