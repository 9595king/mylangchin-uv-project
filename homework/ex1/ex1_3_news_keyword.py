from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. 예시 데이터 (요구사항 2 – 최소 3개)
examples = [
    {
        "news": "삼성전자가 내년 초에 자체적으로 개발한 인공지능(AI) 가속기를 처음으로 출시할 예정이다. "
                "이는 AI 반도체 시장에서 지배적인 위치를 차지하고 있는 엔비디아의 독점을 도전하고, "
                "세계 최고의 반도체 제조업체로서의 지위를 다시 확립하려는 삼성전자의 노력으로 해석된다.",
        "keywords": "삼성전자, 인공지능, 엔비디아",
    },
    {
        "news": "세계보건기구(WHO)는 최근 새로운 건강 위기에 대응하기 위해 국제 협력의 중요성을 강조했다. "
                "전염병 대응 역량의 강화와 글로벌 보건 시스템의 개선이 필요하다고 발표했다.",
        "keywords": "세계보건기구, 건강위기, 국제협력",
    },
    {
        "news": "한국 축구 국가대표팀이 월드컵 아시아 지역 예선에서 승리를 거두며 본선 진출에 한 걸음 더 다가섰다. "
                "대표팀은 이날 경기에서 두 골을 넣으며 조 1위 자리를 지켰다.",
        "keywords": "축구국가대표팀, 월드컵예선, 본선진출",
    },
]

# 2. 예시 1개를 대화 형태로 바꾸는 틀
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{news}"),
    ("ai", "키워드: {keywords}"),
])

# 3. Few-Shot 프롬프트 (요구사항 1)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 4. 최종 프롬프트 = system + 예시 대화들 + 실제 질문
final_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 뉴스 키워드 추출 전문가입니다. "
     "뉴스 기사에서 가장 핵심적인 키워드 정확히 3개를 추출하세요. "
     "반드시 '키워드: 키워드1, 키워드2, 키워드3' 형식으로만 답하고 다른 설명은 하지 마세요."),
    few_shot_prompt,
    ("human", "{input}"),
])

# 5. 모델 & 체인 (요구사항 4 – 일관성 위해 temperature=0)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = final_prompt | llm | StrOutputParser()

# 6. 다양한 분야 뉴스로 테스트 (요구사항 3, 5)
test_news = [
    # IT (지문 테스트 뉴스)
    "제미나이 2.0 플래시는 현재 구글 AI 스튜디오(Google AI Studio) 및 버텍스 AI(Vertex AI)에서 "
    "제미나이 API를 통해 개발자에게 실험 모델로 제공됩니다. 모든 개발자는 멀티모달 입력 및 텍스트 출력을 "
    "사용할 수 있으며, 텍스트 음성 변환(text-to-speech) 및 네이티브 이미지 생성은 일부 파트너들을 대상으로 "
    "제공됩니다. 내년 1월에는 더 많은 모델 사이즈와 함께 일반에 공개될 예정입니다.",
    # 경제
    "한국은행이 기준금리를 동결했다. 물가 상승세가 둔화되고 있지만 가계부채 증가와 환율 변동성을 "
    "고려해 신중한 결정을 내렸다는 분석이 나온다.",
    # 환경
    "올여름 기록적인 폭염이 이어지면서 전력 사용량이 역대 최고치를 기록했다. "
    "전문가들은 기후변화로 인한 이상기온이 앞으로 더 잦아질 것이라고 경고했다.",
]

for i, news in enumerate(test_news, start=1):
    result = chain.invoke({"input": news})
    print(f"[뉴스 {i}] {news[:30]}...")
    print(result)
    print("-" * 50)