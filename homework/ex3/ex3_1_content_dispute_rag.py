import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage, UpstageEmbeddings

# .env 파일을 불러와서 환경 변수로 설정
load_dotenv()

# ChatUpstage, UpstageEmbeddings 가 UPSTAGE_API_KEY 환경 변수를 자동으로 읽음 (키 값은 출력하지 않음)
if not os.getenv("UPSTAGE_API_KEY"):
    raise RuntimeError(".env 파일에 UPSTAGE_API_KEY 가 없습니다.")

# 경로 설정 (.py 를 프로젝트 루트에서 uv run 으로 실행하므로 루트 기준 경로)
PDF_PATH = "data/콘텐츠분쟁해결_사례.pdf"
DB_PATH = "db/faiss_db_content_dispute"  # FAISS 인덱스 저장 폴더 (분할 설정을 바꾸면 이 폴더를 지우고 다시 실행)

# 0단계: 문서 로드
print("==> 0. 문서 로딩 → PDF 읽기...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()  # 페이지 1장 = Document 1개 (metadata 에 page, page_label 포함)
print(f"  총 {len(documents)}페이지 로드 완료")

# 1단계: 문서 분할 설정
print("==> 1. 문서 분할 → 작은 청크로 나누기")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,        # 법률 사례는 1200-1800자 권장
    chunk_overlap=300,      # 사례 맥락 보존을 위해 200-400자
    separators=[
        "\n【사건개요】",     # 법률 문서 섹션 구분자
        "\n【쟁점사항】",     # 쟁점 부분 구분
        "\n【처리경위】",     # 처리 과정 구분
        "\n【처리결과】",     # 결과 부분 구분
        "\n■", "\n\n", "\n", ".", " ", ""
    ],
)
chunks = text_splitter.split_documents(documents)  # Document 리스트를 나누므로 split_documents (metadata 유지)
print(f"  {len(chunks)}개 청크 생성 완료")

# 2단계: 임베딩 모델 설정
print("==> 2. 벡터화 → 임베딩 모델 준비")
embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

# 2-1단계: FAISS 벡터스토어 — 저장된 인덱스가 있으면 불러오고, 없으면 만들어 저장 (매번 임베딩 API 호출 방지)
if (Path(DB_PATH) / "index.faiss").exists():
    print(f"==> 2-1. 기존 FAISS 인덱스 로드: {DB_PATH}")
    # 직접 만든 인덱스만 로드할 것 (pickle 사용)
    vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("==> 2-1. 저장 → FAISS 벡터스토어 생성")
    vectorstore = FAISS.from_documents(chunks, embeddings)  # 청크마다 임베딩 API 를 호출해 벡터로 저장
    vectorstore.save_local(DB_PATH)
    print(f"  FAISS 인덱스 저장 완료: {DB_PATH} ({len(chunks)}개 벡터)")

# 3단계: 검색기 설정
retriever = vectorstore.as_retriever(
    search_type="similarity",        # 또는 "mmr" (다양성 고려시)
    search_kwargs={"k": 5}           # 상위 5개 관련 사례 검색
)

# 4단계: LLM 설정
llm = ChatUpstage(
    model="solar-pro3",
    base_url="https://api.upstage.ai/v1",
    temperature=0.2                  # 사례에 충실한 답변을 위해 낮게
)

# 5단계: 법률 자문 프롬프트 작성 (문자열 안에는 # 주석을 넣지 않음 — 그대로 LLM 에 전달되기 때문)
prompt_template = """
당신은 콘텐츠 분야 전문 법률 자문가입니다.
아래 분쟁조정 사례들을 바탕으로 정확하고 전문적인 법률 조언을 제공해주세요.

관련 분쟁사례:
{context}

상담 내용: {question}

답변 가이드라인:
1. 제시된 사례들을 근거로 답변하세요
2. 관련 법령이나 조항이 있다면 명시하세요
3. 비슷한 사례의 처리경위와 결과를 참고하여 설명하세요
4. 실무적 해결방안을 단계별로 제시하세요
5. 사례에 없는 내용은 "제시된 사례집에서는 확인할 수 없습니다"라고 명시하세요

전문 법률 조언:"""

prompt = PromptTemplate.from_template(prompt_template)  # 문자열 → 체인에 넣을 수 있는 프롬프트 객체


def format_docs(docs):
    """검색된 문서들을 하나의 문자열로 포맷 (사례 사이를 빈 줄로 구분)"""
    return "\n\n".join(doc.page_content for doc in docs)


# 6단계: LCEL 체인 생성
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}  # 질문 하나를 두 갈래로 보냄
    | prompt
    | llm
    | StrOutputParser()
)


# 8단계: 분쟁 유형 분류 함수 (선택)
def classify_dispute_type(query):
    game_keywords = ["게임", "아이템", "계정", "캐릭터", "레벨", "길드", "온라인게임"]
    elearning_keywords = ["강의", "온라인교육", "이러닝", "수강", "환불", "화상교육"]
    web_keywords = ["웹사이트", "무료체험", "자동결제", "구독", "사이트"]

    query_lower = query.lower()

    if any(keyword in query_lower for keyword in game_keywords):
        return "게임"
    elif any(keyword in query_lower for keyword in elearning_keywords):
        return "이러닝"
    elif any(keyword in query_lower for keyword in web_keywords):
        return "웹콘텐츠"
    else:
        return "기타"


# 7단계: 테스트 질문
test_questions = [
    "온라인 게임에서 시스템 오류로 아이템이 사라졌는데, 게임회사가 복구를 거부하고 있습니다. 어떻게 해결할 수 있나요?",
    "인터넷 강의를 중도 해지하려고 하는데 과도한 위약금을 요구받고 있습니다. 정당한가요?",
    "무료체험 후 자동으로 유료전환되어 요금이 청구되었습니다. 환불 가능한가요?",
    "미성년자가 부모 동의 없이 게임 아이템을 구매했습니다. 환불받을 수 있는 방법이 있나요?",
    "온라인 교육 서비스가 광고와 다르게 제공되어 계약을 해지하고 싶습니다. 가능한가요?",
]

print("\n" + "=" * 60)
print(" 콘텐츠분쟁해결 RAG 시스템 테스트")
print("=" * 60)

for i, question in enumerate(test_questions, 1):
    dispute_type = classify_dispute_type(question)
    print(f"\n【테스트 {i}/{len(test_questions)}】 [분쟁 유형: {dispute_type}]")
    print(f"Q: {question}")
    try:
        answer = qa_chain.invoke(question)  # 체인 맨 앞 dict 가 질문 문자열 하나를 받아 context/question 으로 나눔
        print(f"A: {answer}")
    except Exception as e:
        # API 키 오류, 네트워크 오류 등이 나도 다음 질문은 계속 테스트
        print(f"  오류 발생: {e}")
    print("-" * 40)