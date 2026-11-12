import os
from pathlib import Path
import pickle

# Streamlit Cloud(리눅스 헤드리스 환경)에서 Orange3 임포트 시 Qt 디스플레이 오류 방지
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import Orange
import streamlit as st

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 위험도 기준 상수 (필요시 임계값 조정 가능)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="교통사고 사망자 예측 및 어린이 보호구역 안전 대시보드",
    page_icon="🚦",
    layout="wide",
)

SAFE_THRESHOLD = 10.0     # 10명 미만: 🟢 안전
CAUTION_THRESHOLD = 20.0  # 10명 이상 ~ 20명 미만: 🟡 주의 / 20명 이상: 🔴 위험

MODEL_PATH = Path(__file__).parent / "traffic_casualty_model.pkcls"


# -----------------------------------------------------------------------------
# 2. 오렌지3 모델 로드 함수 (캐싱 적용)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_orange_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict_fatalities(
    model,
    sido: str,
    accident_count: float,
    minor_injury_count: float,
    severe_injury_count: float = np.nan,
    reported_injury_count: float = np.nan,
) -> float:
    """오렌지3 모델 도메인에 맞춰 입력 Table을 생성하고 예상 사망자 수를 반환합니다."""
    orig_domain = getattr(model, "original_domain", model.domain)
    input_attrs = list(orig_domain.attributes)
    input_domain = Orange.data.Domain(input_attrs)

    value_map = {
        "시도": sido,
        "사고건수": float(accident_count),
        "중상자수": float(severe_injury_count),
        "경상자수": float(minor_injury_count),
        "부상신고자수": float(reported_injury_count),
    }

    row = [value_map.get(attr.name, np.nan) for attr in input_attrs]
    input_table = Orange.data.Table.from_list(input_domain, [row])

    prediction = model(input_table)
    pred_value = float(prediction[0])
    return max(0.0, pred_value)  # 음수 예측 방지


# -----------------------------------------------------------------------------
# 3. 커스텀 CSS (대형 신호등 및 안전 수칙 카드 스타일)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .traffic-box {
        background-color: #1e222a;
        border-radius: 28px;
        padding: 28px 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 28px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
        margin: 12px 0 20px 0;
    }
    .lamp {
        width: 96px;
        height: 96px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.15rem;
        color: rgba(255,255,255,0.35);
        transition: all 0.25s ease;
        border: 4px solid #111318;
    }
    .lamp-red-off { background-color: #4a1c1c; }
    .lamp-yellow-off { background-color: #4a3f18; }
    .lamp-green-off { background-color: #184227; }

    .lamp-red-on {
        background-color: #ff3b30;
        color: #ffffff;
        box-shadow: 0 0 35px #ff3b30, 0 0 70px rgba(255, 59, 48, 0.7);
        transform: scale(1.08);
    }
    .lamp-yellow-on {
        background-color: #ffcc00;
        color: #1a1a1a;
        box-shadow: 0 0 35px #ffcc00, 0 0 70px rgba(255, 204, 0, 0.7);
        transform: scale(1.08);
    }
    .lamp-green-on {
        background-color: #34c759;
        color: #ffffff;
        box-shadow: 0 0 35px #34c759, 0 0 70px rgba(52, 199, 89, 0.7);
        transform: scale(1.08);
    }
    .status-banner {
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 18px;
        border-left: 8px solid;
    }
    .safety-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    }
    .safety-card h4 {
        margin-top: 0;
        color: #0f172a;
    }
    .safety-item {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
        border-left: 4px solid #3b82f6;
        color: #1e293b;
        font-size: 0.96rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 4. 메인 UI 및 실시간 예측 실행
# -----------------------------------------------------------------------------
st.title("🚦 교통사고 예상 사망자 수 예측 및 어린이 보호구역 안전 대시보드")
st.caption("Orange3 신경망 회귀 모델(`traffic_casualty_model.pkcls`) 기반 실시간 시뮬레이터")

try:
    model = load_orange_model(MODEL_PATH)
except Exception as e:
    st.error(f"모델 로드 중 오류가 발생했습니다: {e}")
    st.info("GitHub 저장소 루트 폴더(`app.py`와 같은 위치)에 `traffic_casualty_model.pkcls` 파일이 있는지 확인해주세요.")
    st.stop()

# 모델 도메인에서 시도 목록 자동 추출 (기본값: ['경기', '부산', '서울'])
orig_domain = getattr(model, "original_domain", model.domain)
if "시도" in orig_domain and hasattr(orig_domain["시도"], "values"):
    sido_options = list(orig_domain["시도"].values)
else:
    sido_options = ["서울", "경기", "부산"]

left_col, right_col = st.columns([1, 1.35], gap="large")

with left_col:
    st.subheader("1. 사고 조건 입력")
    selected_sido = st.selectbox(
        "시도 선택",
        options=sido_options,
        index=sido_options.index("서울") if "서울" in sido_options else 0,
        help="모델이 학습한 시도 권역을 선택하세요.",
    )

    accident_count = st.slider(
        "사고건수 (건)",
        min_value=0,
        max_value=5000,
        value=1000,
        step=10,
        help="슬라이더를 움직이면 예상 사망자 수가 실시간으로 재계산됩니다.",
    )

    minor_injury_count = st.slider(
        "경상자수 (명)",
        min_value=0,
        max_value=7000,
        value=1200,
        step=10,
        help="슬라이더를 움직이면 예상 사망자 수가 실시간으로 재계산됩니다.",
    )

    with st.expander("고급 입력: 중상자수 · 부상신고자수 (선택사항)"):
        use_custom_extra = st.checkbox(
            "중상자수 및 부상신고자수 직접 지정 (체크 해제 시 모델 학습 평균값 자동 적용)",
            value=False,
        )
        if use_custom_extra:
            severe_injury_count = st.slider("중상자수 (명)", 0, 2000, 250, step=5)
            reported_injury_count = st.slider("부상신고자수 (명)", 0, 1000, 100, step=5)
        else:
            severe_injury_count = np.nan
            reported_injury_count = np.nan

# 슬라이더/셀렉트박스 값이 바뀔 때마다 즉시 예측 실행
predicted_deaths = predict_fatalities(
    model=model,
    sido=selected_sido,
    accident_count=accident_count,
    minor_injury_count=minor_injury_count,
    severe_injury_count=severe_injury_count,
    reported_injury_count=reported_injury_count,
)

# 위험 단계 판정
if predicted_deaths < SAFE_THRESHOLD:
    level = "SAFE"
    status_emoji = "🟢"
    status_label = "안전 (SAFE)"
    status_desc = f"예상 사망자 수가 {SAFE_THRESHOLD:.0f}명 미만으로 비교적 안정적인 수준입니다. 기본 안전 수칙을 준수하세요."
    banner_bg = "#ecfdf5"
    banner_border = "#10b981"
    banner_text = "#065f46"
elif predicted_deaths < CAUTION_THRESHOLD:
    level = "CAUTION"
    status_emoji = "🟡"
    status_label = "주의 (CAUTION)"
    status_desc = f"예상 사망자 수가 {SAFE_THRESHOLD:.0f}~{CAUTION_THRESHOLD:.0f}명 구간입니다. 스쿨존 감속 및 보행자 보호에 각별히 주의하세요."
    banner_bg