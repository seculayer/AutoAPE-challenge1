# *Spooky Author Identification*

### 목적
트레이닝 셋에 주어진 각 구문별 작가 라벨을 학습하여 테스트에서 주어진 구문이 어떤 작가의 것일지 각각의 확률 계산

### 진행 방식
1. 라벨 벡터화
2. 데이터 전처리 (특수 기호 살림)
3. N(bi)-gram 사용
4. GridSearchCV와 FastText를 활용한 KerasClassifier 모델 활용
