# ReadMe
[toc]



# Pipeline

데이터 전처리 모듈로, 모든 데이터를 하나의 피클로 모아주는 기능을 수행함. 이 첼린지에서는 `baseline`데이터와 `derived`데이터, 그리고 `GnssLog`데이터를 포함한다. 또한 학습 데이터의 경우에는 `ground_truth`또한 포함한다.

`Pipeline`에서는 `Wrangling`과 `EDA`를 수행한다.



# Deeplearning

## Baseline

GoogleSmartphoneDecimalChallenge(GSDC) 데이터를 불러오고, 처리하여, 제출파일을 생성하는 과정을 하는 가장 기본적인 뼈대의 코드이다. 또한 학습 과정에서 가장 좋은 학습 데이터에 대한 checkpoint를 생성해주고, 매 학습마다 하이퍼 파라미터와 학습 결과등을 저장하여 언제든 재사용 가능하게 만든다.

### 트라이

3000개짜리 임시 버퍼에 두고 3000 X features 의 크기를 갖는 데이터 처리하드시 해보면 될까?



## Modeling





# KalmanFilter

아얘 접근법을 달리해서 칼만필터 기반으로 점수를 올려본 다음 생각해보자.

coordinate frame을 어떻게 가져가야 할까?

1. ENU-Frame: Local 좌표계로 변환해 잘 알고 있는 dynamics로 모델링 쌉가능
2. WGS84(Longtitude, Latitude?): 변환할 필요는 없지만, 연산이 복잡할 것 같음







# Must Remove



## Baseline

기본적인 학습을 위한 베이스 라인 코드
가장 간단하게 MLP형태로 작성

## Pipeline
데이터 가공 파이프라인(추후 파이썬 코드로 수정)
모델에서 학습할 수 있도록 데이터들을 변환 가공해주는 역할을 수행함

데이터들은 시계열 데이터에 가깝고, collectName과 phoneName단위로 그루핑 할 수 있을 것으로 보임
따라서 동일한 collectName, phoneName에서는 i번째 데이터 추정을 위해 [i - WINDOW_SIZE, i]만큼의 궤적정보를 활용

데이터를 다룰때 모든 데이터를 한군데 모아서 정리한 뒤 활용하는 방식은 데이터가 무겁기 때문에 활용하기 어려운 방식처럼 보임

~~따라서 어차피 시계열 데이터라고 한다면 시계열 묶음은 (collectionName, phoneName)을 하나의 단위로 끊고, 그에 맞는 데이터를 필요할때마다 load하는 방식은 어떨까?~~

PICKLE로 묶어서 사용하면 상대적으로 빠르게 활용 가능해보임()