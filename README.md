# Simple-NSMC-Sentiment-Analysis

참고
https://ai.shop2world.net/nsmcnaver-sentiment-movie-corpus-%eb%8d%b0%ec%9d%b4%ed%84%b0%ec%85%8b-%ec%9d%b4%ec%9a%a9-%ea%b0%90%ec%84%b1-%eb%b6%84%eb%a5%98sentiment-analysis-%eb%ac%b8%ec%a0%9c-%ed%95%b4%ea%b2%b0/

# NSMC 데이터셋을 이용한 감성 분류 문제 해결
## 큰 흐름

1. 데이터 불러오기: 'pandas' 라이브러리를 사용하여 훈련 및 테스트 데이터를 불러옵니다.

2. 데이터 전처리: null 값이 있는 행을 제거합니다.
3. 단어장 생성: 'CountVectorizer'를 사용하여 텍스트 데이터를 수치 데이터로 변환하기 위한 단어장을 생성합니다.

4. 모델 훈련:
Logistic Regression 모델을 훈련합니다.
훈련 데이터를 벡터화하고 레이블을 준비하여 모델을 학습시킵니다.
5. 테스트 및 예측:
테스트 데이터를 벡터화하고 학습된 모델을 사용하여 예측합니다.
6. 사용자 입력 처리:
사용자로부터 영화 리뷰를 입력받고, 이를 모델에 입력하기 위해 벡터화합니다.
7. 결과 출력:
모델이 예측한 결과를 확인하여 긍정적인 리뷰 또는 부정적인 리뷰로 출력합니다.
