from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. 프롬프트 (요구사항 1, 2)
prompt = PromptTemplate.from_template(
    """당신은 집에 있는 재료로 요리를 추천해 주는 친절한 AI 요리사입니다.
사용자가 알려준 재료로 만들 수 있는 요리 1가지를 추천하고, 간단한 레시피를 알려주세요.

재료: {ingredients}

아래 형식으로 답변해 주세요.

추천 요리: (요리 이름)
재료: (사용하는 재료)
조리법:
1. ...
2. ...
3. ...
"""
)

# 2. 모델 (요구사항 3)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 3. 파서 (요구사항 4)
parser = StrOutputParser()

# 4. 체인 (요구사항 5)
chain = prompt | llm | parser

# 5. 실행
result = chain.invoke({"ingredients": "계란, 밥, 김치"})
print(result)