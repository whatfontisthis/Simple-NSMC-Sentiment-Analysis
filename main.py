path_to_ratings_train = "/content/drive/MyDrive/NSMC/ratings_train.txt"
path_to_ratings_test = "/content/drive/MyDrive/NSMC/ratings_train.txt"

# 필요한 라이브러리 가져오기
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# 데이터 불러오기
# 'pandas' 라이브러리를 사용하여 데이터를 불러옵니다.
# 'path_to_ratings_test'와 'path_to_ratings_train'은 데이터 파일의 경로입니다.
train_data = pd.read_csv(path_to_ratings_train, sep='\t', na_values='NaN')
test_data = pd.read_csv(path_to_ratings_test, sep='\t', na_values='NaN')

# null 값 제거
# 누락된 데이터를 처리하기 위해 null 값을 제거합니다.
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# CountVectorizer를 이용하여 단어장 생성
# 'CountVectorizer'는 텍스트 데이터를 수치 데이터로 변환하기 위해 사용됩니다.
# 'token_pattern=r'\b\w+\b''는 단어를 토큰화할 때 단어 경계를 인식하도록 설정합니다.
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
vectorizer.fit(train_data['document'])  # 단어장 생성을 위해 훈련 데이터를 사용합니다.

# train 데이터를 이용하여 Logistic Regression 모델 생성
# 'Logistic Regression'은 이진 분류 모델을 만들기 위한 알고리즘입니다.
# 데이터를 벡터화하고 이 모델을 훈련시킵니다.
lr = LogisticRegression(max_iter=5000)
train_X = vectorizer.transform(train_data['document'])  # 텍스트 데이터를 벡터로 변환
train_y = train_data['label']  # 긍정/부정 레이블
lr.fit(train_X, train_y)  # 모델 학습

# test 데이터를 이용하여 예측 수행
test_X = vectorizer.transform(test_data['document'])  # 테스트 데이터를 벡터로 변환
pred_y = lr.predict(test_X)  # 테스트 데이터 예측

# 사용자로부터 입력 받은 문장 예측 수행
user_input = input("영화 리뷰를 입력하세요: ")
input_X = vectorizer.transform([user_input])  # 입력 문장을 벡터로 변환
result = lr.predict(input_X)  # 입력 데이터 예측

# 예측 결과 출력
# 모델이 예측한 결과를 출력합니다.
if result[0] == 1:
    print("긍정적인 리뷰입니다. ")
else:
    print("부정적인 리뷰입니다. ")
