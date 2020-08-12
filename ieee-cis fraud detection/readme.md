# *IEEE-CIS Fraud Detection*

### 목적
거래 내역을 보고 진짜 사람이 사용한 것이 맞는지 도용(fraud) 여부 판단

### 진행 방식
#### 학습 데이터 전처리 (데이터 많이 날아감)
1. identity 데이터와 transaction 데이터 merge (공통된 ID가 없는 행 삭제)
2. null 값이 80% 이상인 칼럼과 테스트 데이터에 없는 칼럼 삭제
3. Imputer 이용한 빈 항목 채움

GradientBoostingClassifier 이용
