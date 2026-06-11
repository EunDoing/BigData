# Titanic Feature Engineering Pipeline

빅데이터분석 과제 3

Titanic 데이터셋을 활용하여 다양한 Feature Engineering 기법을 적용하고 머신러닝 모델의 성능을 비교하였다.

## 주요 내용

- 결측치 처리 (Mean, Median, Most Frequent)
- 범주형 변수 인코딩 (One-Hot Encoding, Label Encoding)
- 스케일링 (StandardScaler, MinMaxScaler, RobustScaler)
- 파생 변수 생성
  - family_size
  - is_alone
  - age_group
  - fare_log
- Feature Selection (SelectKBest)
- 모델 비교
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- SHAP 기반 변수 중요도 분석

## 결과

- 최고 ROC-AUC: Exp-1 + Logistic Regression (0.8495)
- 최고 Accuracy: Base + LightGBM (0.8268)

## Repository Structure
