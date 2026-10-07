from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

# TODO 1. 1단계 프롬프트 – 여행지 → 명소 이름
prompt1 = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 세계 여행 전문가입니다. "
     "사용자가 알려준 여행지의 대표 관광 명소 1곳을 추천하세요. "
     "설명이나 문장 없이 명소 이름만 한 단어로 출력하세요."),
    ("user", "{city}의 대표 관광 명소 1곳을 추천해 주세요."),
])

# TODO 2. 2단계 프롬프트 – 명소 → 상세 정보
prompt2 = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 친절한 여행 가이드입니다. "
     "관광 명소에 대해 아래 형식으로 한국어로 설명하세요.\n\n"
     "명소: (명소 이름)\n\n"
     "역사:\n1. ...\n2. ...\n\n"
     "특징:\n1. ...\n2. ...\n\n"
     "방문 팁:\n1. ...\n2. ..."),
    ("user", "{place}에 대한 역사, 특징, 방문 팁을 알려주세요."),
])

# TODO 3. 단계별 체인
chain1 = prompt1 | llm | parser
chain2 = prompt2 | llm | parser

# TODO 4. 두 체인 연결 + 중간 결과 보존
full_chain = (
    RunnablePassthrough.assign(place=chain1)
    | RunnablePassthrough.assign(detail=chain2)
)

# TODO 5. 실행 및 출력
result = full_chain.invoke({"city": "로마"})
print("1단계 결과 (추천 명소):", result["place"])
print("=" * 50)
print("2단계 결과 (상세 정보):")
print(result["detail"])