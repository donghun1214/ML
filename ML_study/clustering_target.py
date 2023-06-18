#군집 알고리즘 : 1. 타깃 데이터 (이미지) 각 부부 픽셀 값들을 구한다.

# 2. 2차원으로 표시된 픽셀값들과의 오차평균이 작은 순서대로 나열한다. 그 나열된 리스트들이 답.


!wget https://bit.ly/fruits_300_data -O fruits_300.npy


import numpy as np

import matplotlib.pyplot as plt 


fruits = np.load('fruits_300.npy')

plt.imshow(fruits[0], cmap = 'gray')


fig, axs = plt.subplots(1,2) #1행 2열 로 subplot 만듬. axs 는 subplot 객체

axs[0].imshow(fruits[100], cmap = 'gray_r') #1행 1열 

axs[1].imshow(fruits[200], cmap = 'gray_r') #1행 2열 

plt.show()


#픽셀 값 분석해서 apple, banna, pineapple 구별하기 

apple = fruits[0:100].reshape(-1, 100 * 100) # 계산(mean 계산)하기 용이하도록 1차원으로 만듬.

pineapple = fruits[100:200].reshape(-1, 100*100)

banana = fruits[200:300].reshape(-1, 100*100)


apple_mean = np.mean(apple, axis = 0).reshape(100,100)

pineapple_mean = np.mean(pineapple, axis = 0).reshape(100,100)

banana_mean = np.mean(banana, axis = 0).reshape(100,100)


abs_diff = np.abs(fruits - apple_mean)

abs_mean = np.mean(abs_diff, axis = (1,2)) #axis 1 방향 (행), 2방향(열) 이므로 2차원 전체 오차 평균이다. 


apple_index = np.argsort(abs_mean)[:100] #argsort : 작은 순으로 정렬된 index값 반환

fig, axs = plt.subplots(10,10, figsize= (10,10))

for i in range(10):

for j in range(10):

axs[i,j].imshow(fruits[apple_index[i*10 + j]], cmap = 'gray_r')

axs[i,j].axis('off')

plt.show()



