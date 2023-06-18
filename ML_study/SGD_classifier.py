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


import numpy as np

from sklearn.linear_model import SGDClassifier

#손실함수 logistic 함수로 지정.

#L = -(target * log(a) + (1- target) * log(1 - a)) * a = 예측 확률 

sc = SGDClassifier(loss = 'log_loss', random_state = 42)

train_score = []

test_score = []

classes = np.unique(train_target) #부분 학습(partial_fit)는 전체 클래스가 무엇이 있는지 명시해줘야 됨.

#epoch에 따른 최적의 max_iter을 찾아줘야 됨

for _ in range(0, 300):

sc.partial_fit(train_scaled, train_target, classes = classes)

train_score.append(sc.score(train_scaled, train_target))

test_score.append(sc.score(test_scaled, test_target))


import matplotlib.pyplot as plt

plt.plot(train_score)

plt.plot(test_score)

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.show()

#최적의 max_iter = 100


sc = SGDClassifier(loss = 'log_loss', max_iter = 100, tol = None, random_state = 42)

sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))

print(sc.score(test_scaled, test_target))


#손실함수 logistic 말고도 hinge 손실함수로도 쓰임

