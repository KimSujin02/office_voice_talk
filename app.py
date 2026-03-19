import os
import re
import uuid
import tempfile
from html import escape
from konlpy.tag import Okt

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from openai import OpenAI
from faster_whisper import WhisperModel
from transformers import pipeline
import pyttsx3

from dotenv import load_dotenv

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
# =========================================================
# 기본 설정
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="직장 커뮤니케이션 시뮬레이션 음성 챗봇",
    page_icon="💼",
    layout="wide"
)

st.title("💼 사회초년생 직장 커뮤니케이션 시뮬레이션 음성 챗봇")
st.caption("상황/상대/특징을 설정하고 음성으로 대화를 연습한 뒤, 호감도 점수와 피드백을 받습니다.")


# =========================================================
# 환경 변수
# =========================================================
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")
HF_SENTIMENT_MODEL = os.getenv("HF_SENTIMENT_MODEL")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "small")

# =========================================================
# OpenAI 클라이언트
# =========================================================
@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# =========================================================
# Whisper 로드
# =========================================================
@st.cache_resource
def load_whisper_model():
    """
    faster-whisper 모델 로드
    - CPU 기준으로 base/small 추천
    - compute_type=int8 은 CPU에서 무난
    """
    model = WhisperModel(
        WHISPER_SIZE,
        device="cpu",
        compute_type="int8"
    )
    return model


# =========================================================
# Hugging Face 감정분석 로드
# =========================================================
@st.cache_resource
def load_sentiment_pipeline():
    clf = pipeline(
        task="text-classification",
        model=HF_SENTIMENT_MODEL,
        tokenizer=HF_SENTIMENT_MODEL,
        truncation=True
    )
    return clf


# =========================================================
# Okt 형태소 분석기 로드 (사용자 패턴 분석용)
# =========================================================
@st.cache_resource
def load_okt():
    return Okt()

# =========================================================
# 세션 상태 초기화
# =========================================================
def init_session_state():
    defaults = {
        "simulation_started": False,
        "simulation_ended": False,
        "simulation": {
            "situation": "",
            "target_role": "",
            "target_traits": [],
            "user_role": "",
            "goal": "",
            "max_turns": 5,
            "current_turn": 0,
            "score": 50
        },
        "chat_history": [],
        "turn_scores": [],
        "user_analysis": {
            "empathy_count": 0,
            "question_count": 0,
            "polite_count": 0,
            "short_reply_count": 0,
            "positive_word_count": 0,
            "total_reply_length": 0,
            "reply_count": 0,
            "noun_count_total": 0,
            "verb_count_total": 0,
            "reaction_count": 0
        },
        "last_transcript": "",
        "last_ai_reply": "",
        "last_tts_audio": None,
        "last_sentiment": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_simulation():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()


init_session_state()


# =========================================================
# 유틸
# =========================================================
def clamp_score(score: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, score))


def format_chat_history_for_prompt(chat_history: list) -> str:
    if not chat_history:
        return "이전 대화 없음"

    lines = []
    for msg in chat_history:
        speaker = msg.get("speaker", "AI" if msg["role"] == "assistant" else "나")
        lines.append(f"{speaker}: {msg['text']}")
    return "\n".join(lines)


def build_system_prompt(simulation: dict) -> str:
    traits_text = ", ".join(simulation["target_traits"]) if simulation["target_traits"] else "특징 없음"

    return f"""
당신은 직장 커뮤니케이션 시뮬레이션용 챗봇입니다.

[현재 역할 설정]
- 상황: {simulation["situation"]}
- 당신 역할: {simulation["target_role"]}
- 당신 특징: {traits_text}
- 사용자 역할: {simulation["user_role"]}
- 사용자 목표: {simulation["goal"]}

[규칙]
1. 항상 설정된 당신 역할에 맞게 대답하세요.
2. 직장 내 실제 대화처럼 자연스럽고 현실적으로 말하세요.
3. 매 답변은 1~3문장으로 간결하게 하세요.
4. 이전 대화 내용을 기억하고 이어가세요.
5. 사용자의 답변에 따라 호감/불편함/중립적인 반응이 자연스럽게 드러나게 하세요.
6. 당신의 역할의 특징({traits_text})이 자연스럽게 반영되게 하세요.
""".strip()


# =========================================================
# STT
# =========================================================
def transcribe_audio_file(uploaded_audio_file) -> str:
    """
    Streamlit의 st.audio_input()가 반환한 audio를
    faster-whisper로 텍스트로 변환하여 반환
    """
    model = load_whisper_model()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_audio_file.getvalue())
        tmp_path = tmp.name

    try:
        segments, _info = model.transcribe(
            tmp_path,
            language="ko",
            vad_filter=True
        )

        text_parts = []
        for segment in segments:
            seg_text = segment.text.strip()
            if seg_text:
                text_parts.append(seg_text)

        transcript = " ".join(text_parts).strip()
        return transcript
    # finally:
    #     try:
    #         os.remove(tmp_path)
    except OSError:
        pass


# =========================================================
# LLM: OpenAI
# =========================================================
def generate_ai_reply(user_input: str, simulation: dict, chat_history: list) -> str:
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    system_prompt = build_system_prompt(simulation)
    history_text = format_chat_history_for_prompt(chat_history)

    user_prompt = f"""
[이전 대화]
{history_text}

[현재 사용자 입력]
{user_input}

[현재 호감도 점수]
{simulation["score"]}

위 정보를 반영하여 다음 상대 대사를 생성하세요.
""".strip()

    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.8,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()


# =========================================================
# TTS: pyttsx3
# =========================================================
def synthesize_speech(text: str) -> bytes:
    """
    pyttsx3로 wav 생성 후 bytes 반환
    """
    engine = pyttsx3.init()

    try:
        # 속도 / 볼륨 조절
        current_rate = engine.getProperty("rate")
        engine.setProperty("rate", max(120, current_rate - 20))
        engine.setProperty("volume", 1.0)

        # 한국어 음성 찾기 시도
        voices = engine.getProperty("voices")
        selected_voice_id = None

        for voice in voices:
            voice_text = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
            if "korean" in voice_text or "ko_" in voice_text or "kr" in voice_text:
                selected_voice_id = voice.id
                break

        if selected_voice_id:
            engine.setProperty("voice", selected_voice_id)

        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")

        engine.save_to_file(text, temp_path)
        engine.runAndWait()

        with open(temp_path, "rb") as f:
            audio_bytes = f.read()

        return audio_bytes

    finally:
        try:
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass

        try:
            engine.stop()
        except Exception:
            pass


# =========================================================
# 감정분석: Hugging Face
# =========================================================
def normalize_sentiment_label(raw_label: str) -> str:
    label = str(raw_label).strip().lower()

    mapping = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
        "1 star": "negative",
        "2 stars": "negative",
        "3 stars": "neutral",
        "4 stars": "positive",
        "5 stars": "positive"
    }

    if label in mapping:
        return mapping[label]

    negative_keywords = ["분노", "화남", "불안", "슬픔", "걱정", "짜증", "부정", "실망", "당황"]
    neutral_keywords = ["중립", "보통", "무감정", "평온"]
    positive_keywords = ["기쁨", "행복", "감사", "호감", "만족", "긍정", "편안", "반가움"]

    if any(k in raw_label for k in negative_keywords):
        return "negative"
    if any(k in raw_label for k in neutral_keywords):
        return "neutral"
    if any(k in raw_label for k in positive_keywords):
        return "positive"

    return "neutral"


def sentiment_to_score(label: str, confidence: float) -> int:
    base_map = {
        "positive": 6,
        "neutral": 0,
        "negative": -6
    }
    base = base_map.get(label, 0)

    if confidence >= 0.90:
        weight = 1.4
    elif confidence >= 0.75:
        weight = 1.2
    elif confidence >= 0.60:
        weight = 1.0
    else:
        weight = 0.8

    return int(round(base * weight))


def analyze_ai_sentiment(ai_reply: str) -> dict:
    clf = load_sentiment_pipeline()
    result = clf(ai_reply)[0]

    raw_label = result["label"]
    confidence = float(result["score"])
    normalized = normalize_sentiment_label(raw_label)
    delta = sentiment_to_score(normalized, confidence)

    return {
        "raw_label": raw_label,
        "label": normalized,
        "confidence": confidence,
        "score_delta": delta
    }


# =========================================================
# 사용자 패턴 분석
# =========================================================
def analyze_user_pattern(user_text: str):
    analysis = st.session_state.user_analysis
    okt = load_okt()

    text = user_text.strip()
    if not text:
        return

    # 형태소 / 품사 분석
    morphs = okt.morphs(text)
    pos_tags = okt.pos(text)

    # 규칙 사전
    empathy_words = {
        "그렇군요", "그랬군요", "이해", "공감", "고생", "힘드시", "힘드셨",
        "바쁘셨", "대단", "정말요", "맞아요"
    }

    polite_words = {
        "감사합니다", "죄송합니다", "알겠습니다", "괜찮습니다",
        "맞습니다", "확인했습니다", "부탁드립니다", "실례지만"
    }

    positive_words = {
        "좋", "괜찮", "감사", "다행", "반갑", "재미", "든든", "편하", "잘"
    }

    # 추가 통계용 키가 없으면 만들어줌
    if "noun_count_total" not in analysis:
        analysis["noun_count_total"] = 0
    if "verb_count_total" not in analysis:
        analysis["verb_count_total"] = 0
    if "reaction_count" not in analysis:
        analysis["reaction_count"] = 0

    # 1. 공감 표현
    if any(word in text for word in empathy_words):
        analysis["empathy_count"] += 1

    # 2. 예의 표현
    if any(word in text for word in polite_words):
        analysis["polite_count"] += 1

    # 3. 질문 여부
    # 문장부호 + 종결 패턴 + 형태소 일부 조합
    question_patterns = ["?", "까요", "나요", "으세요", "세요", "어떠세요", "맞나요"]
    has_question = any(p in text for p in question_patterns)

    # 품사 기반 보조 판정
    # Okt pos 예시: [('알려', 'Verb'), ('주', 'Verb'), ('세요', 'Eomi')]
    if not has_question:
        pos_joined = [f"{m}/{p}" for m, p in pos_tags]
        if any(p.endswith("/Eomi") for p in pos_joined) and any(q in text for q in ["까요", "나요", "세요"]):
            has_question = True

    if has_question:
        analysis["question_count"] += 1

    # 4. 긍정 단어 수
    positive_count = sum(text.count(word) for word in positive_words)
    analysis["positive_word_count"] += positive_count

    # 5. 리액션 표현
    reaction_words = {"네", "아", "오", "정말", "헉", "와", "아하", "그렇군요"}
    if any(word in text for word in reaction_words):
        analysis["reaction_count"] += 1

    # 6. 단답 판정
    # 너무 짧거나, 형태소 수가 매우 적으면 단답으로 간주
    if len(text) <= 10 or len(morphs) <= 3:
        analysis["short_reply_count"] += 1

    # 7. 대화 확장성용 통계
    noun_count = sum(1 for _, tag in pos_tags if tag == "Noun")
    verb_count = sum(1 for _, tag in pos_tags if tag in ["Verb", "Adjective"])

    analysis["noun_count_total"] += noun_count
    analysis["verb_count_total"] += verb_count

    # 8. 전체 길이 / 횟수
    analysis["total_reply_length"] += len(text)
    analysis["reply_count"] += 1

# =========================================================
# 피드백 텍스트 생성
# =========================================================
def get_feedback_text():
    analysis = st.session_state.user_analysis

    reply_count = analysis["reply_count"]
    avg_len = 0 if reply_count == 0 else analysis["total_reply_length"] / reply_count
    avg_noun = 0 if reply_count == 0 else analysis["noun_count_total"] / reply_count
    avg_verb = 0 if reply_count == 0 else analysis["verb_count_total"] / reply_count

    strengths = []
    improvements = []

    if analysis["polite_count"] >= 2:
        strengths.append("예의 표현이 비교적 안정적으로 사용되었습니다.")
    else:
        improvements.append("직장 상황에서는 예의 표현을 조금 더 자주 넣는 편이 좋습니다.")

    if analysis["question_count"] >= 2:
        strengths.append("질문을 통해 대화를 자연스럽게 이어가려는 시도가 좋았습니다.")
    else:
        improvements.append("짧은 질문을 섞으면 대화가 덜 끊기고 더 자연스럽습니다.")

    if analysis["empathy_count"] >= 1:
        strengths.append("상대 말에 공감하는 표현이 포함되어 좋은 인상을 줄 수 있습니다.")
    else:
        improvements.append("공감 표현이 부족해 다소 사무적으로 느껴질 수 있습니다.")

    if analysis["short_reply_count"] >= 3:
        improvements.append("단답형 응답이 많아 대화가 빨리 끝날 가능성이 있습니다.")
    else:
        strengths.append("답변이 지나치게 짧지 않아 대화 유지에 도움이 됩니다.")

    if avg_len >= 15:
        strengths.append("평균 답변 길이가 비교적 적절했습니다.")
    else:
        improvements.append("답변을 한 문장만 더 확장하면 더 자연스럽게 들립니다.")

    if avg_noun >= 2:
        strengths.append("대화 소재가 될 만한 핵심 표현을 비교적 잘 포함했습니다.")
    else:
        improvements.append("구체적인 소재나 키워드를 조금 더 넣으면 대화가 풍부해집니다.")

    if avg_verb >= 1.5:
        strengths.append("행동이나 상태를 설명하는 표현이 적절히 포함되었습니다.")
    else:
        improvements.append("의견이나 상태를 드러내는 표현을 조금 더 넣으면 자연스럽습니다.")

    return strengths, improvements


# =========================================================
# 시각화
# =========================================================
def draw_pattern_chart():
    analysis = st.session_state.user_analysis

    data = {
        "공감 표현": analysis["empathy_count"],
        "질문 횟수": analysis["question_count"],
        "예의 표현": analysis["polite_count"],
        "단답 횟수": analysis["short_reply_count"],
        "긍정 단어": analysis["positive_word_count"],
        "리액션": analysis["reaction_count"]
    }

    df = pd.DataFrame({
        "항목": list(data.keys()),
        "값": list(data.values())
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["항목"], df["값"])
    ax.set_title("사용자 대화 패턴 분석")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20)

    st.pyplot(fig)


# =========================================================
# 채팅 렌더
# =========================================================
def render_chat():
    st.subheader("💬 대화 내역")

    for msg in st.session_state.chat_history:
        safe_text = escape(msg["text"]).replace("\n", "<br>")
        speaker = escape(msg.get("speaker", "AI"))

        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style='text-align:right; margin:10px 0;'>
                    <span style='background-color:#DCF8C6; padding:10px 14px; border-radius:12px; display:inline-block; max-width:75%;'>
                        <b>나</b><br>{safe_text}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            score_delta = msg.get("score_delta")
            score_text = ""

            if score_delta is not None:
                sign = "+" if score_delta > 0 else ""
                score_text = f"<br><small>점수 변화: {sign}{score_delta}</small>"

            st.markdown(
                f"""
                <div style='text-align:left; margin:10px 0;'>
                    <span style='background-color:#F1F0F0; padding:10px 14px; border-radius:12px; display:inline-block; max-width:75%;'>
                        <b>{speaker}</b><br>{safe_text}{score_text}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# 턴 처리
# =========================================================
def process_turn(audio_file, fallback_text: str = ""):
    sim = st.session_state.simulation

    if audio_file is None and not fallback_text.strip():
        st.warning("음성 입력 또는 텍스트 입력이 필요합니다.")
        return

    with st.spinner("음성 인식 → AI 응답 생성 → 감정 분석 → 음성 합성 중..."):
        # STT
        if audio_file is not None:
            user_text = transcribe_audio_file(audio_file)
        else:
            user_text = fallback_text.strip()

        # 만들지 못한 경우
        if not user_text:
            st.warning("음성을 텍스트로 인식하지 못했습니다. 다시 시도해주세요.")
            return

        st.session_state.last_transcript = user_text

        # 사용자 채팅 히스토리 저장
        st.session_state.chat_history.append({
            "role": "user",
            "speaker": "나",
            "text": user_text
        })

        # 사용자 패턴 분석
        analyze_user_pattern(user_text)

        # LLM 응답 생성
        ai_reply = generate_ai_reply(
            user_input=user_text,
            simulation=sim,
            chat_history=st.session_state.chat_history
        )
        st.session_state.last_ai_reply = ai_reply

        # AI 답장의 감정 분석 + 점수 반영
        sentiment = analyze_ai_sentiment(ai_reply)
        st.session_state.last_sentiment = sentiment

        sim["score"] = clamp_score(sim["score"] + sentiment["score_delta"])
        
        # AI 답장 저장
        st.session_state.chat_history.append({
            "role": "assistant",
            "speaker": sim["target_role"],
            "text": ai_reply,
            "score_delta": sentiment["score_delta"]
        })

        st.session_state.turn_scores.append({
            "turn": sim["current_turn"],
            "emotion": sentiment["label"],
            "raw_label": sentiment["raw_label"],
            "confidence": sentiment["confidence"],
            "score_delta": sentiment["score_delta"],
            "total_score": sim["score"]
        })

        # ai 답변으로 TTS 생성
        try:
            tts_audio = synthesize_speech(ai_reply)
            st.session_state.last_tts_audio = tts_audio
        except Exception:
            st.session_state.last_tts_audio = None

        # 턴 종료 처리
        if sim["current_turn"] >= sim["max_turns"]:
            st.session_state.simulation_ended = True
        else:
            sim["current_turn"] += 1


# =========================================================
# 사이드바
# =========================================================
with st.sidebar:
    st.header("⚙️ 시뮬레이션 설정")

    situation = st.selectbox(
        "상황 선택",
        ["회식", "점심 후 티타임", "업무 보고", "엘리베이터에서 마주침", "야근 중 대화", "첫 출근 인사"]
    )

    target_role = st.selectbox(
        "상대 역할",
        ["팀장님", "직속 상사", "옆자리 선배", "부장님", "인사 담당자", "동료"]
    )

    target_traits_input = st.text_input(
        "상대 특징 입력",
        placeholder="예: 엄격함, 친절함, 무뚝뚝함, 유머러스함 등"
    )

    # 쉼표 기준으로 리스트 변환
    target_traits = [
        trait.strip()
        for trait in target_traits_input.split(",")
        if trait.strip()
    ]

    user_role = st.selectbox(
        "나의 역할",
        ["인턴", "신입사원"]
    )

    goal = st.selectbox(
        "대화 목표",
        ["좋은 인상 남기기", "무난하게 대화 이어가기", "실수하지 않기", "친근감 얻기"]
    )

    max_turns = st.slider("대화 턴 수", 3, 10, 5)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        if st.button("시뮬레이션 시작", use_container_width=True):
            reset_simulation()

            st.session_state.simulation_started = True
            st.session_state.simulation_ended = False
            st.session_state.simulation = {
                "situation": situation,
                "target_role": target_role,
                "target_traits": target_traits,
                "user_role": user_role,
                "goal": goal,
                "max_turns": max_turns,
                "current_turn": 1,
                "score": 50
            }

            opening_message = (
                f"{target_role}과 {situation} 상황입니다. 자연스럽게 이야기해 봅시다."
            )

            st.session_state.chat_history.append({
                "role": "assistant",
                "speaker": target_role,
                "text": opening_message
            })

    with c2:
        if st.button("초기화", use_container_width=True):
            reset_simulation()
            st.rerun()


# =========================================================
# 메인 화면
# =========================================================
if not st.session_state.simulation_started:
    st.info("왼쪽 사이드바에서 설정 후 시뮬레이션을 시작해주세요.")

else:
    sim = st.session_state.simulation

    top_left, top_right = st.columns([2, 1])

    with top_left:
        st.subheader("📌 현재 시뮬레이션 정보")
        st.write(f"**상황:** {sim['situation']}")
        st.write(f"**상대 역할:** {sim['target_role']}")
        st.write(f"**상대 특징:** {', '.join(sim['target_traits']) if sim['target_traits'] else '없음'}")
        st.write(f"**나의 역할:** {sim['user_role']}")
        st.write(f"**대화 목표:** {sim['goal']}")
        st.write(f"**현재 턴:** {sim['current_turn']} / {sim['max_turns']}")

    with top_right:
        st.metric("현재 호감도 점수", sim["score"])

    render_chat()

    if not st.session_state.simulation_ended:
        st.subheader("🎤 음성 입력")
        audio_file = st.audio_input("마이크로 녹음하세요", key=f"audio_turn_{sim['current_turn']}")

        st.subheader("또는 텍스트 테스트")
        fallback_text = st.text_area(
            "마이크가 안 되면 텍스트로 입력",
            key=f"text_turn_{sim['current_turn']}",
            height=100
        )

        if st.button("이번 턴 처리", type="primary", use_container_width=True):
            try:
                process_turn(audio_file=audio_file, fallback_text=fallback_text)
                st.rerun()
            except Exception as e:
                st.error(f"처리 중 오류가 발생했습니다: {e}")

        if st.session_state.last_transcript:
            st.info(f"최근 STT 결과: {st.session_state.last_transcript}")

        if st.session_state.last_tts_audio:
            st.audio(st.session_state.last_tts_audio, format="audio/wav")

        if st.session_state.last_sentiment:
            latest = st.session_state.last_sentiment
            st.write("### 최근 감정분석 결과")
            st.write(
                f"- 라벨: **{latest['label']}**\n"
                # f"- 원본 라벨: `{latest['raw_label']}`\n"
                # f"- confidence: `{latest['confidence']:.4f}`\n"
                f"- 점수 변화: `{latest['score_delta']:+d}`"
            )

    # 턴 종료됨.
    else:
        st.success("시뮬레이션이 종료되었습니다.")

        final_score = sim["score"]
        if final_score >= 80:
            grade = "매우 긍정적"
        elif final_score >= 60:
            grade = "긍정적"
        elif final_score >= 40:
            grade = "보통"
        else:
            grade = "개선 필요"

        st.subheader("🏁 최종 결과")
        st.metric("최종 호감도 점수", final_score)
        st.write(f"**최종 평가:** {grade}")

        # draw_pattern_chart()

        strengths, improvements = get_feedback_text()

        st.subheader("✅ 강점")
        if strengths:
            for item in strengths:
                st.write(f"- {item}")
        else:
            st.write("- 아직 뚜렷한 강점 분석 데이터가 부족합니다.")

        st.subheader("⚠️ 개선 포인트")
        if improvements:
            for item in improvements:
                st.write(f"- {item}")
        else:
            st.write("- 전반적으로 무난한 대화 흐름을 보였습니다.")

        # 앞으로 대화 내용과 패턴 분석 데이터를 활용해 더 구체적이고 맞춤화된 피드백과 예시 답변을 추가할 수 있습니다.
        # AI로 분석해서 사용자 맞춤으로 보여주는 방향으로 확장할 예정
        
        # st.subheader("💡 추천 답변 예시")
        # st.write("- 상대가 개인 이야기를 꺼냈다면 짧게 공감한 뒤 가벼운 질문으로 이어가면 좋습니다.")
        # st.write("- 예: “정말요? 많이 바쁘셨겠네요. 요즘은 좀 괜찮으세요?”")

        if st.session_state.last_tts_audio:
            st.subheader("🔊 마지막 AI 응답 음성")
            st.audio(st.session_state.last_tts_audio, format="audio/wav")

        if st.button("다시 시작하기", use_container_width=True):
            reset_simulation()
            st.rerun()