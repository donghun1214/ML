#로지스틱 : 선형방적식을 이용한다. 독립변수들을 이용해 종속변수 Z값을 이끌어내고, binary 냐 multi 냐에 따라 sigmoid 함수, softmax 함수로 갈린다.
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')


#dataframe 에서 여러 개를 column-indexing 할 때 이중 대괄호 써줘야 된다

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()

fish_target = fish['Species'].to_numpy()


from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state =42)


#이것도 역시 표준화 전처리 과정이 필요 

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

ss.fit(train_input)

train_scaled = ss.transform(train_input)

test_scaled = ss.transform(test_input)


#로지스틱 회귀 - binary

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')

train_bream_smelt = train_scaled[bream_smelt_indexes]

target_bream_smelt = train_target[bream_smelt_indexes]


from sklearn.linear_model import LogisticRegression 

lr = LogisticRegression()

lr.fit(train_bream_smelt, target_bream_smelt)


print(lr.predict(train_bream_smelt[:5]))

# sigmoid func 로 결과를 나타낸다.

print(lr.predict_proba(train_bream_smelt[:5]))


#로지스틱 회귀 내부 

decisions = lr.decision_function(train_bream_smelt[:5])

from scipy.special import expit

print(expit(decisions)) # decision_function 값을 시그모이드 함수로 나타낸 것.


#로지스틱 회귀 - multi class


#C는 규제 강도, 

lr = LogisticRegression(C = 20, max_iter = 1000)

lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))

print(lr.score(test_scaled, test_target))


#로지스틱 회귀 내부

from scipy.special import softmax

import numpy as np

decision = lr.decision_function(test_scaled[:5])

proba = softmax(decision, axis = 1) # 여러 z 값을 넣어서 확률을 나타내야 하므로 softmax 가 적절.

print(np.round(proba, decimals = 3))



