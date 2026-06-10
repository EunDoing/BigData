빅데이터분석 01분반

[과제3] 
Feature Engineering 파이프라인 구현 및 성능 비교 실험 보고서

AI학과 2443765 은수인
GitHub: https://github.com/EunDoing/BigData

1. 데이터셋 소개
Kaggle Titanic 데이터셋을 사용하였다. 타이타닉 승객 891명의 정보를 담고 있으며, 생존 여부(survived)를 예측하는 이진 분류 문제이다. 수치형(age, fare 등)과 범주형(sex, embarked 등) 변수가 혼합되어 있으며, age(19.9%)와 deck(77.2%)에 결측치가 존재한다.

컬럼 설명: survived(생존 여부, 타겟), pclass(객실 등급 1/2/3), sex(성별), age(나이, 결측 19.9%), sibsp(동승 형제·배우자 수), parch(동승 부모·자녀 수), fare(운임, 이상치 다수), embarked(탑승 항구 C/Q/S), deck(갑판, 결측 77.2%)

2. EDA 결과
생존율은 38.4%로 약간의 클래스 불균형이 존재한다. pclass와 survived 간 음의 상관관계(-0.34)가 확인되었고, 여성의 생존율이 남성보다 현저히 높았다. fare 컬럼에서는 1등석을 중심으로 극단적 이상치가 다수 발견되었으며, deck 컬럼은 결측률이 너무 높아 'Unknown'으로 통합 처리하였다.

3. Feature Engineering 과정
결측치 처리는 mean, median, most_frequent 세 가지 전략을 비교하였다.인코딩은 One-Hot Encoding과 Label(Ordinal) Encoding 두 가지를 비교하였다. Exp-2에서 Label Encoding을 적용하였으며, 범주 수가 적은 Titanic 데이터에서는 두 방식의 성능 차이가 미미하였다. 스케일링은 StandardScaler, MinMaxScaler, RobustScaler를 비교하였다. 파생 변수는 총 4개를 생성하였는데, sibsp와 parch를 합산한 family_size, 혼자 탑승 여부를 나타내는 is_alone, 나이대를 구간화한 age_group, fare의 이상치 완화를 위한 fare_log(로그 변환)이다.

4. 모델 학습 과정
Logistic Regression, Random Forest, XGBoost, LightGBM 4개 모델을 Base, Exp-1, Exp-2, Exp-3 총 4가지 실험 조건에서 비교하였다. Exp-3(Most Frequent + RobustScaler + Feature Selection) 조건에서 전반적으로 가장 높은 성능이 나타났으며, 최적 조합인 Exp-3 + LightGBM에 GridSearchCV를 추가 적용하여 하이퍼파라미터를 튜닝하였다.

5. 성능 비교 결과
실험 조건별로 비교하면 Base는 전처리 없이 기본 성능을 측정하였고, Exp-1은 Mean 대치와 StandardScaler 적용으로 Logistic Regression 성능이 뚜렷하게 향상되었다. Exp-2는 Label(Ordinal) Encoding과 MinMaxScaler, Feature Selection을 적용하였으며, Exp-3는 RobustScaler가 fare의 이상치에 강건하게 작용하여 전체 모델에서 가장 안정적인 성능을 보였다. Tree 계열 모델(RF, XGBoost, LightGBM)은 스케일링에 민감하지 않았으나, Logistic Regression은 StandardScaler 적용 시 성능 향상이 두드러졌다.

6. 최종 결론
가장 효과적인 전처리 전략은 Exp-3(Most Frequent + RobustScaler + Feature Selection)이었다. One-Hot Encoding은 범주 수가 적은 Titanic 데이터에 적합하였다. Feature Selection은 불필요한 변수를 제거하여 모델 단순화에 기여하였고, ROC-AUC 성능도 유지되었다. SHAP 분석 결과 sex, fare_log, pclass, age가 예측에 가장 큰 영향을 미치는 변수로 확인되었으며, fare_log 파생 변수가 이상치 완화와 예측력 향상에 동시에 기여함을 확인하였다.
