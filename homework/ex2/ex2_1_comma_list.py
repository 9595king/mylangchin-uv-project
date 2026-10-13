from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from llm_config import get_llm  # 공통 LLM 설정 (수업의 get_llm 셀)

# 콤마로 구분된 리스트 출력 파서 초기화
output_parser = CommaSeparatedListOutputParser()

# 출력 형식 지침 가져오기 ("쉼표로 구분해서 답하라"는 안내문)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

# 프롬프트 템플릿 설정 ({interest}는 실행할 때마다, {format_instructions}는 미리 채움)
prompt = PromptTemplate(
    template="{interest} 분야와 관련된 한국의 유명한 장소나 활동 5가지를 추천해 주세요.\n{format_instructions}",
    input_variables=["interest"],
    partial_variables={"format_instructions": format_instructions},
)

# LLM 설정 (추천이라 약간의 다양성 허용)
llm = get_llm(temperature=0.7)

# 프롬프트, 모델, 출력 파서를 연결하여 체인 생성
chain = prompt | llm | output_parser

# 사용자 입력을 받아 동적으로 처리
interest = input("관심 분야를 입력하세요 (예: 음식, 스포츠, 영화): ")
result = chain.invoke({"interest": interest})

# 쉼표로 구분된 리스트 출력
print(f"{interest} 관련 추천 목록:")
print(result)
print(type(result))  # <class 'list'> 확인