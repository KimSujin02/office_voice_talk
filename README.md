# 💼 직장 커뮤니케이션 시뮬레이션 음성 챗봇

> 사회초년생을 위한 **직장 대화 연습 AI 시뮬레이터**  
> 음성 기반으로 대화를 진행하고, AI가 상대 역할을 수행하며  
> 감정 분석과 사용자 패턴 분석을 통해 피드백을 제공하는 시스템

---

## 📌 프로젝트 목적

사회초년생은 회식, 티타임, 업무 보고 등 다양한 직장 상황에서 대화 방식에 대한 부담을 느끼는 경우가 많다.
본 프로젝트는 이러한 상황을 가상으로 재현하고, 사용자가 음성으로 대화를 연습할 수 있도록 하며, AI가 상대방 역할을 수행하면서 대화의 흐름과 반응을 분석하여 커뮤니케이션 역량 향상에 도움을 주는 것을 목적으로 한다.

* 직장 내 대화 상황 사전 연습 가능
* 상대 반응 기반 피드백 제공
* 자신의 대화 패턴을 객관적으로 확인 가능
* 음성 인터페이스를 활용한 몰입형 시뮬레이션 경험 제공

---

## ⚙️ 기술 스택

### 🖥️ Frontend
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### 🧠 AI / NLP
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4.1--mini-412991?style=for-the-badge&logo=openai&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![KoNLPy](https://img.shields.io/badge/KoNLPy-Okt-2C2C2C?style=for-the-badge)

### 🎤 음성 처리
![Whisper](https://img.shields.io/badge/faster--whisper-STT-0A0A0A?style=for-the-badge)
![pyttsx3](https://img.shields.io/badge/pyttsx3-TTS-4CAF50?style=for-the-badge)

### 📊 데이터 처리
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

### ⚙️ 환경
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🎯 주요 기능

### 1. 시뮬레이션 설정 기능

사용자가 대화 시작 전 시뮬레이션 조건을 설정한다.

### 설정 항목

* 대화 턴 수
* 상황
* 상대 역할
* 상대 특징
* 사용자 역할

### 예시

* 상황: 회식
* 상대: 옆자리 선배
* 특징: 공감을 좋아함
* 사용자 역할: 신입사원
* 대화 턴 수: 5턴

---

## 2. 음성 기반 대화 기능

사용자의 음성을 입력받아 STT로 텍스트로 변환한 뒤, AI가 설정된 상대 역할로 답변한다.

### 처리 흐름

1. 사용자 음성 입력
2. STT 변환
3. 이전 대화 기록과 현재 설정을 포함해 LLM 호출
4. AI 응답 생성
5. 응답 TTS 재생
6. 채팅 내역 화면 출력

---

## 3. 대화 기억 기능

AI는 이전 대화를 기억하여 맥락이 이어지는 멀티턴 대화를 수행한다.

### 기억 대상

* 이전 사용자 발화
* 이전 AI 발화
* 상대가 언급한 정보
* 감정 흐름
* 현재 점수 상태

---

## 4. 감정/반응 분석 기능

매 턴마다 AI의 응답을 분석하여 사용자의 대화가 상대에게 어떤 반응을 이끌어냈는지 점수화한다.

### 점수 이름 추천

**호감도 점수**

### 점수 기준 예시

* 매우 긍정적 반응: +10
* 긍정적 반응: +5
* 중립적 반응: 0
* 부정적 반응: -5
* 매우 부정적 반응: -10

---

## 5. 결과 리포트 기능

대화 종료 후 최종 점수와 함께 사용자의 답변 패턴을 시각적으로 분석하여 보여준다.

### 출력 항목

* 최종 호감도 점수
* 턴별 점수 변화 그래프
* 답변 패턴 분석
* 강점 및 개선점
* 추천 답변 예시

---

# 🤖 기능 명세서

| 기능명      | 설명               | 입력                  | 출력         |
| -------- | ---------------- | ------------------- | ---------- |
| 시뮬레이션 설정 | 대화 조건 설정         | 상황, 상대, 특징, 역할, 턴 수 | 설정 완료 정보   |
| 음성 입력    | 사용자 음성 수집        | 음성 파일/마이크 입력        | 음성 데이터     |
| STT 변환   | 음성을 텍스트로 변환      | 음성 데이터              | 사용자 텍스트    |
| 대화 생성    | AI가 상대 역할로 응답    | 사용자 텍스트, 설정, 대화기록   | AI 응답 텍스트  |
| 기억 관리    | 이전 대화 맥락 유지      | 대화 히스토리             | 누적 컨텍스트    |
| 감정 분석    | AI 응답 감정 및 반응 분석 | AI 응답 텍스트           | 감정 레이블, 점수 |
| TTS 변환   | AI 응답을 음성으로 변환   | AI 응답 텍스트           | 음성 출력      |
| 채팅 UI 출력 | 대화 내역 시각화        | 사용자/AI 텍스트          | 채팅창 표시     |
| 결과 분석    | 전체 대화 리포트 생성     | 전체 대화 로그            | 점수/그래프/피드백 |

---
# 🖼️ 시스템 흐름도

```text
[사용자 설정 입력]
    ↓
[시뮬레이션 시작]
    ↓
[사용자 음성 입력]
    ↓
[STT 변환]
    ↓
[대화 히스토리 + 설정 + 현재 상태 기반 프롬프트 생성]
    ↓
[LLM 응답 생성]
    ↓
[응답 감정 분석 및 점수 계산]
    ↓
[TTS 음성 출력]
    ↓
[채팅창 반영 + 점수 표시]
    ↓
[남은 턴 있으면 반복]
    ↓
[최종 결과 분석 및 시각화]
```
