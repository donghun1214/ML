import pandas as pd

df = pd.read_csv('https://bit.ly/perch_csv_data') #csv 파일로 받아서 pandas에서 dataframe형태로 바꿈.

perch_full = df.to_numpy() #numpy 형태로 바꿈

print(perch_full)


import numpy as np

perch_weight = np.array(

[5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 

130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 

514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 

1000.0, 1000.0])


from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state = 42)


#다항 회귀를 위한 데이터 전처리 과저

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()

poly.fit(train_input) #훈련 데이터로 학습한 것으로 train, test 모두 적용(transform)

train_poly = poly.transform(train_input)

test_poly = poly.transform(test_input)


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(train_input, train_target)


#특성이 많으면, 과대적합이 일어남 --> 규제가 있는 알고리즘 필요


#릿지, 라쏘(규제 회구) 하기 전에 표준화해야 됨.

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

ss.fit(train_poly)

train_scaled = ss.transform(train_poly)

test_scaled = ss.transform(test_poly)


#규제 : ridge 회귀

from sklearn.linear_model import Ridge

ridge = Ridge()

ridge.fit(train_scaled, train_target)

print(ridge.score(test_scaled, test_target))


import matplotlib.pyplot as plt

train_score = []

test_score = []

#alpha 값은 규제의 완화정도를 나타낸다. 최적의 alpha 값 찾기.

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:

ridge = Ridge(alpha = alpha)

ridge.fit(train_scaled, train_target)

train_score.append(ridge.score(train_scaled, train_target))

test_score.append(ridge.score(test_scaled, test_target))


plt.plot(np.log10(alpha_list), train_score)

plt.plot(np.log10(alpha_list), test_score)

plt.xlabel('alpha')

plt.ylabel('R^2')

plt.show()


ridge = Ridge(alpha = 0.1)

ridge.fit(train_scaled, train_target)

print(ridge.score(train_scaled, train_target))

print(ridge.score(test_scaled, test_target))

