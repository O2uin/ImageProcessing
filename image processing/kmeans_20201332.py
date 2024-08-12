import numpy as np
import matplotlib.pyplot as plt
import cv2

#이미지에서 데이터 불러오기
img=cv2.imread('sample.jpg')
img=cv2.resize(img,(480,480))

data=img.reshape((-1,3))
data=np.float32(data)

#클러스터 개수 설정
K=2

#라벨은 라이브러리 기능을 이용해 가져왔습니다
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
ret, label, center_=cv2.kmeans(data,K,None,criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


#data값 label별로 분류해 저장
def make_mapping(data):
    data_=[]
    for i in range(K):
        data_.append(data[np.where(label==i)[0]])
    return data_

data_=make_mapping(data)


#평균 계산
def calc_k_means(data_):
    means=[np.mean(data_[k],axis=0) for k in range(K)]
    return means

means=calc_k_means(data_)


#중심 클러스터 업데이트
def update_k(data,means,label):
    n=[]
    for p in data:
        dists=[np.linalg.norm(means[k]-p)for k in range(K)]
        label=np.argmin(dists)

update_k(data, means, label)

#반복작업
def fit(data, epochs=10):
    for e in range(epochs):
        data_=make_mapping(data)
        means=calc_k_means(data_)
        update_k(data,means,label)
    return means, data_

means, data_=fit(data)

#화면 그리기
center=np.uint8(means)
res=center[label.flatten()]

res=res.reshape(img.shape)

cv2.imshow('original',img)
cv2.imshow('KMEANS',res)

cv2.waitKey()
cv2.destroyAllWindows()
