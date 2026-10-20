# 🚑 ICU 급성 신손상(AKI) 조기 예측 및 중증도 분류

### MIMIC-IV 기반 Prediction Cutoff 설계와 2단계 머신러닝 모델 개발

> 미니 프로젝트에서 발견한 데이터 전처리 및 모델 검증의 한계를 바탕으로, 예측 시점을 명확하게 정의하고 AKI 발생 예측에서 중증도 분류까지 확장한 프로젝트입니다.

**Tech Stack**  
`Python` `SQL` `Pandas` `Scikit-learn` `Logistic Regression` `LightGBM` `XGBoost` `SHAP` `Optuna`

**Project Type:** Team Project  
**Dataset:** MIMIC-IV  
**Previous Project:** [AKI Early Warning Mini Project](https://github.com/leechaehui/AKI_early_warning_miniproject)

---

## 01. Project Overview | 프로젝트 개요

### 프로젝트 배경

급성 신손상(Acute Kidney Injury, AKI)은 신장 기능이 급격하게 저하되는 상태로, 중환자실(ICU) 환자의 예후에 중요한 영향을 미치는 합병증입니다.

기존 AKI 진단에 활용되는 혈청 크레아티닌(Creatinine)은 신장 기능의 변화를 즉각적으로 반영하지 못할 수 있어, 다양한 임상 데이터를 활용한 조기 위험 예측이 중요합니다.

앞선 미니 프로젝트에서는 MIMIC-IV 데이터를 활용하여 AKI 발생 예측 모델을 구축했지만, 데이터 처리 방식과 모델 평가 과정에서 추가 검증이 필요한 부분을 확인했습니다.

이에 최종 프로젝트에서는 **Prediction Cutoff를 기준으로 관찰구간과 예측구간을 분리하고, AKI 발생 여부뿐 아니라 중증도까지 분류하는 2단계 예측 모델**을 개발했습니다.

### 프로젝트 목표

- ICU 환자의 향후 48시간 이내 AKI 발생 위험 예측
- KDIGO 기준 기반 AKI 중증도 분류
- Prediction Cutoff 기반 Feature 및 Label 구간 설계
- 임상적 의미를 고려한 Feature Engineering
- 머신러닝·딥러닝 모델 성능 비교
- 클래스 불균형과 Stage 간 분포 중첩 문제 개선
- CDSS 활용 가능성 검토

---

## 02. Project Development | 미니 프로젝트에서 최종 프로젝트로의 발전

미니 프로젝트에서 확인한 한계점을 바탕으로 분석 과정과 예측 목표를 확장했습니다.

| 구분 | 미니 프로젝트 | 최종 프로젝트 |
|---|---|---|
| 예측 목표 | AKI 발생 여부 예측 | 향후 48시간 AKI 발생 및 중증도 분류 |
| 시간적 설계 | 예측 시점 검증 필요성 확인 | Prediction Cutoff 기준 관찰·예측구간 명시 |
| Feature Engineering | 신기능 및 허혈·관류 변수 설계 | 신기능, 소변량, 활력징후, 혈액검사 등 36개 변수 구성 |
| 데이터 품질 | 결측값 처리 방식의 적절성 검증 필요 | 결측값·이상치 및 입력 변수 검토 |
| 모델 평가 | XGBoost 중심 성능 비교 | 머신러닝·딥러닝 모델 비교 및 단계별 모델 선정 |
| 클래스 불균형 | 클래스 가중치 중심 대응 | Class Weight, SMOTE, Undersampling 등 비교 |
| 예측 구조 | AKI 이진 분류 | 2단계 예측 및 중증도 그룹 재정의 |

### 핵심 개선 방향

**① 예측 시점과 데이터 사용 범위 명확화**

Prediction Cutoff를 기준으로 과거 48시간 데이터만 Feature로 사용하고, 이후 48시간을 예측 및 Label 구간으로 설정했습니다.

**② 다양한 모델 및 검증 전략 비교**

머신러닝과 딥러닝 알고리즘을 비교하고, 성능과 해석 가능성을 고려하여 모델을 선정했습니다.

**③ 분석 결과를 바탕으로 문제 재정의**

Stage 2 분류 성능이 낮게 나타나는 원인을 분석하고, Stage 간 중첩과 클래스 불균형을 고려하여 중증도 분류 구조를 수정했습니다.

---

## 03. Data Pipeline | 데이터 분석 프로세스

### 전체 분석 과정

```text
MIMIC-IV Dataset
        |
        v
01. Cohort Construction
    ICU 분석 대상 환자 선정
        |
        v
02. Prediction Cutoff Definition
    관찰·예측 구간 설정
        |
        v
03. Label Generation
    KDIGO 기반 AKI Stage 라벨 구성
        |
        v
04. Feature Engineering
    36개 임상 변수 구성
        |
        v
05. Data Preprocessing
    결측값·이상치 처리 및 데이터 검증
        |
        v
06. Data Splitting
    Train / Validation / Test 구성
        |
        v
07. Stage 1 Modeling
    AKI 발생 여부 예측
        |
        v
08. Stage 2 Modeling
    AKI 중증도 분류
        |
        v
09. Model Evaluation
    성능 비교 및 오류 분석
        |
        v
10. Model Improvement
    불균형 처리 및 분류 구조 개선
```

---

## 04. Cohort Construction | 코호트 구축

MIMIC-IV 데이터를 기반으로 분석 목적에 적합한 ICU 환자 집단을 선정했습니다.

### 코호트 선정 과정

| 단계 | 선정 기준 | 남은 대상 수 |
|---|---|---:|
| 0 | MIMIC-IV 전체 ICU 입실 기록 | 364,627건 |
| 1 | 성인(18세 이상), ICU 재실 24시간 이상 | 74,829명 |
| 2 | 만성 신질환·투석 관련 ICD 코드 제외 | 56,595명 |
| 3 | 입원 전 반복적인 eGFR 저하 기준 대상 제외 | **42,210명** |

### 최종 코호트

| 항목 | 결과 |
|---|---:|
| 최종 분석 대상 | **42,210명** |
| AKI 발생률 | 25.3% |
| 평균 나이 | 60.8세 |
| 원내 사망률 | 7.4% |

단순히 확보한 모든 환자 데이터를 사용하는 대신, 분석 목적과 신기능 관련 기준을 고려하여 코호트를 구성했습니다.

---

## 05. Prediction Cutoff | 관찰구간 및 예측구간 설계

### 5.1 Prediction Cutoff 기반 시간 구간 설정

본 프로젝트에서는 특정 시점인 **Prediction Cutoff를 기준으로 과거 48시간을 관찰구간, 이후 48시간을 예측구간**으로 설정했습니다.

```text
       Feature Window                  Label Window
       (관찰구간 48h)                   (예측구간 48h)

<--------------------------|-------------------------->
                           |
      과거 임상 데이터      |     향후 AKI 발생 여부
      기반 Feature 생성    |     및 Stage 확인
                           |
                       Prediction
                         Cutoff

     Cutoff - 48h        Cutoff        Cutoff + 48h
```

### 5.2 Feature Window

**Prediction Cutoff -48h ~ Prediction Cutoff**

- 예측 시점 이전 48시간 동안 관측한 데이터 활용
- 소변량, 활력징후, 혈액검사 결과 등으로 Feature 생성
- Prediction Cutoff 이후의 데이터가 입력 변수에 포함되지 않도록 시간적 범위 구분

### 5.3 Label Window

**Prediction Cutoff ~ Prediction Cutoff +48h**

- 향후 48시간 이내 AKI 발생 여부 확인
- KDIGO 기준 기반 AKI Stage 구성
- 모델 학습을 위한 Label 생성

### 설계 의의

예측에 활용할 수 있는 과거 정보와 예측해야 하는 미래 결과를 시간적으로 분리했습니다.

이를 통해 **실제 예측 시점에서 확보할 수 있는 정보만을 사용하는 모델 설계**를 목표로 했습니다.

다만 시간 구간을 분리하는 것과 실제 데이터 누수가 완전히 제거되었다는 것은 다르므로, 변수별 측정 시점 및 전처리 과정의 추가 검증이 중요합니다.

---

## 06. Label & Feature Engineering | 라벨 및 변수 설계

### 6.1 AKI Stage Label

KDIGO 기준을 활용하여 AKI 중증도를 구분했습니다.

| Stage | 환자 수 | 비율 | 구분 |
|---|---:|---:|---|
| Stage 0 | 31,507 | 약 74.6% | Non-AKI |
| Stage 1 | 8,265 | 약 19.6% | 경증 |
| Stage 2 | 1,032 | 약 2.4% | 중등증 |
| Stage 3 | 1,406 | 약 3.3% | 중증 |
| **Total** | **42,210** | **100%** | |

### 데이터 특성

- 전체 환자 중 Non-AKI가 가장 높은 비율을 차지
- Stage 2와 Stage 3의 표본 수가 상대적으로 적음
- 클래스 불균형으로 인해 중증도 분류 성능 저하 가능성 확인

### 6.2 Feature Engineering

Prediction Cutoff 이전 48시간의 임상 데이터를 활용하여 총 **36개 임상 변수**를 구성했습니다.

| Feature Group | 주요 변수 |
|---|---|
| 기본 정보 | 환자 기본 특성 |
| 신기능 | Creatinine, BUN, 소변량 |
| 혈역학 | MAP, HR, 혈압 관련 변수 |
| 약물 관련 | Vasopressor, 승압제 관련 변수 |
| 혈액검사 | Lactate, Potassium, Sodium, Hemoglobin 등 |

### 변수 설계 방향

**① 신기능 관련 지표**

Creatinine과 BUN, 소변량 등을 활용하여 신기능 상태를 반영했습니다.

**② 혈역학적 상태**

MAP, HR, 혈압 관련 지표를 통해 환자의 순환 및 관류 상태를 반영했습니다.

**③ 다양한 임상 정보 통합**

약물 및 혈액검사 관련 변수를 함께 사용하여 다양한 임상 요인을 고려했습니다.

**④ 시간적 적절성**

Prediction Cutoff 이전에 확보 가능한 정보만 활용하도록 Feature 구간을 설정했습니다.

---

## 07. Stage 1 Modeling | AKI 발생 여부 예측

### 모델링 목표

ICU 환자가 Prediction Cutoff 이후 48시간 이내 AKI를 경험하는지 예측하는 이진 분류 모델을 개발했습니다.

### 비교 알고리즘

**Machine Learning**

- Logistic Regression
- Random Forest
- XGBoost

**Deep Learning**

- TabNet
- Time-Series Transformer
- LSTM

### 주요 모델 성능 비교

| Metric | Logistic Regression | LSTM |
|---|---:|---:|
| **AUROC** | **0.8816** | 0.6678 |
| **AUPRC** | **0.7224** | 0.3179 |
| **F1-score** | **0.7799** | 0.1803 |
| **Recall** | **0.9074** | 0.1078 |

### 최종 모델 선정

**Logistic Regression**

Logistic Regression은 비교 결과에서 안정적인 예측 성능을 보였으며, 변수의 영향과 모델 판단 근거를 상대적으로 명확하게 설명할 수 있다는 장점이 있었습니다.

따라서 단순히 복잡한 모델을 적용하는 것보다 **예측 성능, 해석 가능성, 활용 가능성을 함께 고려하여 Logistic Regression을 최종 선정**했습니다.

---

## 08. Stage 2 Modeling | AKI 중증도 분류

### 8.1 초기 중증도 분류

초기에는 AKI 중증도를 Stage 1, Stage 2, Stage 3으로 각각 분류하는 모델을 구성했습니다.

**비교 및 개선 실험**

- LightGBM
- TabNet
- Transformer
- Class Weight
- SMOTE
- Undersampling
- Optuna 기반 하이퍼파라미터 최적화
- 5-Fold Cross Validation
- Threshold 조정 및 Feature Engineering

### 8.2 문제 발견

초기 모델링에서 **Stage 2의 Recall이 약 5% 수준**으로 낮게 나타났습니다.

추가 분석을 통해 다음 문제를 확인했습니다.

**① 클래스 불균형**

Stage 2는 전체 데이터 중 약 2.4%에 해당하여 다른 클래스에 비해 표본 수가 적었습니다.

**② Stage 간 변수 분포 중첩**

Creatinine 분포 비교와 t-SNE 분석에서 Stage 2가 Stage 1과 Stage 3 사이에 위치하는 특성을 확인했습니다.

이로 인해 Stage 2만의 명확한 분류 경계를 형성하기 어려웠습니다.

### 8.3 문제 해결을 위한 실험

| 개선 방법 | 적용 내용 | 결과 |
|---|---|---|
| 모델 비교 | LightGBM, TabNet, Transformer | Stage 2 성능 부족 |
| 불균형 처리 | Class Weight, SMOTE, Undersampling | 개선 효과 제한적 |
| 학습 최적화 | Optuna + 5-Fold CV | 성능 개선 미미 |
| Threshold Tuning | 예측 기준값 조정 | 탐지 증가와 함께 오탐 증가 |
| Feature Engineering | 추가 임상 변수 활용 | 개선 효과 미미 |

15개 이상의 모델 및 개선 전략을 실험했지만, Macro F1-score는 약 0.30 수준에 머물렀고 Stage 2 성능 역시 충분히 개선되지 않았습니다.

### 8.4 문제 재정의

다양한 알고리즘과 불균형 처리 방법을 적용했음에도 Stage 2 분류의 구조적 한계가 지속되었습니다.

이에 기존의 Stage 1, Stage 2, Stage 3 분류를 재검토하고, **Stage 2와 Stage 3을 하나의 고위험군으로 통합**했습니다.

변경된 중증도 분류는 다음과 같습니다.

| 기존 분류 | 개선 후 분류 |
|---|---|
| Stage 1 | Stage 1 |
| Stage 2 | Stage 2+3 (고위험군) |
| Stage 3 | Stage 2+3 (고위험군) |

이는 모델의 성능 수치만 개선하려는 접근이 아니라, **데이터 분포와 임상적 활용 목적을 고려하여 예측 문제 자체를 재정의한 과정**입니다.

---

## 09. Final Model | 최종 모델 구조 및 성능

### 9.1 최종 모델 아키텍처

최종적으로 단계별 특성에 맞는 알고리즘을 적용했습니다.

```text
       ICU Patient Data
               |
               v
     Prediction Cutoff
               |
               v
      Feature Engineering
               |
               v
   [Stage 1: AKI Prediction]
        Logistic Regression
               |
               v
       AKI 발생 여부 예측
               |
               v
   [Stage 2: Severity Classification]
             LightGBM
               |
               v
        AKI 중증도 분류
       Stage 1 / Stage 2+3
```

### 9.2 최종 모델 평가 결과

최종 평가에서는 Non-AKI, Stage 1, Stage 2+3에 대한 분류 성능을 확인했습니다.

| Classification | One-vs-Rest AUROC | Recall |
|---|---:|---:|
| Non-AKI | 0.874 | 79.2% |
| Stage 1 | 0.753 | 32.0% |
| **Stage 2+3** | **0.889** | **77.6%** |

※ 위 AUROC는 최종 3개 클래스에 대한 One-vs-Rest 평가 결과이며, 1단계 이진 분류 모델의 AUROC와 구분됩니다.

### 9.3 결과 해석

**① 고위험군 탐지**

Stage 2+3 통합 고위험군에서 Recall 77.6%, AUROC 0.889를 기록했습니다.

**② Stage 1 분류의 한계**

Stage 1의 Recall은 32.0%로 상대적으로 낮게 나타났습니다. 이는 Stage 간 경계 구분의 어려움이 여전히 존재함을 보여줍니다.

**③ 분석 목표에 맞는 문제 재정의**

Stage 2 분류 성능의 한계를 분석하고, Stage 2·3 통합 구조를 적용하여 중증도 분류 가능성을 검토했습니다.

다만 최종 성능만으로 실제 임상적 활용 가능성이 입증된 것은 아니며, 추가적인 검증이 필요합니다.

---

## 10. My Contribution | 담당 역할

### 10.1 Feature Engineering

- 임상적 의미를 고려한 변수 구성 및 검토
- 허혈·저관류 관련 혈역학 변수 설계
- MAP, Shock Index 등 관련 변수 활용
- 모델 입력 변수의 적절성 검토

### 10.2 Data Preprocessing

- SQL 및 Python 기반 데이터 처리
- 결측값 및 이상치 검토
- 모델 학습용 데이터 구성
- 변수 및 라벨의 시간적 적절성 검토

### 10.3 Machine Learning

- 머신러닝 모델 학습 및 성능 비교
- 하이퍼파라미터 튜닝 및 실험
- 모델 평가 지표 분석
- 모델 성능 저하 원인 검토

### 10.4 Model Interpretation & Validation

- SHAP 기반 변수 중요도 분석
- 데이터 누수 가능성 검토
- 모델 예측 결과의 해석 및 타당성 확인

### 핵심 기여

임상적 의미를 고려한 변수 설계와 데이터 기반 모델링·검증을 통해 AKI 조기 예측 분석에 참여했습니다.

특히 데이터 처리와 평가 과정에서 발생할 수 있는 문제를 검토하고, 성능 수치뿐 아니라 결과의 신뢰성과 해석 가능성을 고려하는 데 중점을 두었습니다.

---

## 11. Limitations | 한계점 및 추가 개선 방향

### 11.1 Stage 1 분류 성능

Stage 2·3을 통합한 이후에도 Stage 1의 Recall은 32.0%로 낮게 나타났습니다.

향후 Stage 간 구분에 영향을 미치는 변수와 데이터 분포를 추가로 분석할 필요가 있습니다.

### 11.2 데이터 품질 및 누수 검증

Prediction Cutoff를 기준으로 Feature Window와 Label Window를 구분했지만, 실제 데이터 추출·변수 생성 과정에서도 시간적 조건이 일관되게 적용되었는지 검증하는 것이 중요합니다.

### 11.3 외부 검증 필요

MIMIC-IV 기반 내부 평가 결과가 다른 의료기관의 환자 데이터에서도 동일하게 유지되는지는 추가 검증이 필요합니다.

### 11.4 임상 적용의 한계

예측 모델을 실제 CDSS에 활용하기 위해서는 모델 성능뿐 아니라 경보 기준, 오탐에 따른 부담, 안전성 등에 대한 추가적인 평가가 필요합니다.

---

## 12. Retrospective | 프로젝트 회고

### 데이터 분석 관점

미니 프로젝트를 통해 머신러닝 모델을 구축한 경험을 바탕으로, 최종 프로젝트에서는 데이터 수집·전처리·예측 시점 정의와 같은 모델 학습 이전 단계의 중요성을 확인했습니다.

### 문제 해결 관점

Stage 2의 분류 성능이 낮게 나타나면서 다양한 알고리즘과 불균형 처리 방법을 비교했습니다.

그러나 기존 접근만으로는 성능 개선에 한계가 있었고, 데이터의 분포와 Stage 간 특성을 분석하여 분류 구조 자체를 변경했습니다.

이를 통해 **모델의 성능이 기대에 미치지 못할 때 알고리즘만 변경하는 것이 아니라, 데이터의 특성과 문제 정의를 함께 검토하는 것이 중요하다는 점**을 배웠습니다.

### 프로젝트를 통해 배운 점

> 모델의 복잡도가 높다고 항상 우수한 예측 성능을 얻는 것은 아니며, 분석 목적과 데이터의 특성에 적합한 방법을 선택하는 것이 중요하다는 점을 배웠습니다.
>
> 특히 예측 시점에 따른 데이터 분리, 클래스 불균형, 중증도 간 분포 중첩 문제를 분석하며 문제를 발견하고 해결 방향을 찾아가는 과정을 경험했습니다.

---

## 13. Project Summary | 프로젝트 요약

| 구분 | 내용 |
|---|---|
| Project | ICU AKI 조기 예측 및 중증도 분류 |
| Dataset | MIMIC-IV |
| Cohort | 42,210명 |
| Feature Window | Prediction Cutoff 이전 48시간 |
| Label Window | Prediction Cutoff 이후 48시간 |
| Feature | 총 36개 임상 변수 |
| Stage 1 Model | Logistic Regression |
| Stage 2 Model | LightGBM |
| Final Classification | Non-AKI / Stage 1 / Stage 2+3 |
| Key Result | Stage 2+3 AUROC 0.889, Recall 77.6% |
| Main Challenge | 클래스 불균형 및 Stage 간 분포 중첩 |
| Improvement | Stage 2·3 통합을 통한 중증도 분류 구조 재정의 |

**핵심 성과**

Prediction Cutoff 기반 시간 구간을 설계하고, AKI 발생 예측 및 중증도 분류 모델을 개발했습니다.

특히 Stage 2 분류 성능이 낮게 나타난 원인을 분석하고 다양한 개선 실험을 수행한 뒤, 데이터의 구조적 특성을 고려하여 예측 문제를 재정의했습니다.

---

**Project Type:** Team Project  
**Focus:** Feature Engineering · Machine Learning · Model Evaluation · Problem Solving  
**Dataset:** MIMIC-IV

*본 프로젝트는 연구 및 학습 목적의 모델 개발 결과이며, 실제 임상 의사결정을 위한 검증된 의료 시스템은 아닙니다.*
