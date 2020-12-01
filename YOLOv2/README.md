
# syim.github.com

PASCAL VOC(Visual Object Classes Challenge) 데이터셋 기반으로 만든 YoloV2 입니다 Convolution network는 Resnet을 사용하였습니다.

Yolo는 V1에서 V2로 버전업 되면서 성능과 속도 모두 개선되었습니다.

- 성능향상요인

![yolov2](https://user-images.githubusercontent.com/44501825/49162393-20614d80-f36e-11e8-9dc4-efcd32b4bd92.jpg)

1. Batch Nomalization : 모든 컨볼루션 레이어에 배치 정규화를 추가 (Dropout을 제거하고 오버피팅 문제를 해결)

2. High Resolution Classifier : iamgeNet의 고해상도 이미지를 Classification network로 학습시켜 고해상도 이미지에서도 잘 동작하게 함.

3. Convolutional : FC(Fully Connected)를 Convolution으로 대체

4. Anchor Boxes : 초기값으로 설정된 Anchor box들로 경계박스를 예측

5. New network : Darknet-19 사용 

6. Dimension Clusters : 실제 학습데이터들의 경계박스들을 K-mean Cluster로 최적의 경게박스들을 찾음(k=5)

7. Direct location prediction : Anchor Box위치의 예측범위를 그리드셀의 범위를 기준으로 하여 위치좌표를 예측

8. passthrough : 28x28의 중간크기 Layer를 스킵하여 13x13 Layer에 concatenate 함 (작은 오브젝트의 검출을 위해 높은 해상도 정보를 포함시킴)

9. Multi-Scale Training : 다양항 해상도 이미지를 입력하여 학습 가능


##  Network Architectures

*YoloV1

![yolov1](https://user-images.githubusercontent.com/44501825/49067351-2ddfe000-f267-11e8-8ce8-1e712c06e346.jpg)

YoloV1의 Convolution Layer 구성을 보면 448x448x3의 Input 이미지가 최종적으로는 FC를 통해 7x7x30의 output을 생성합니다.
여기서 YoloV2와 차이점은 기존 Yolo는 마지막 FC단에서 경계박스를 직접 예측하였지만 V2에서는 FC를 Convloution으로 대체하고
Anchor Box를 이용해 경계 박스를 예측합니다.
그리고 입력 이미지크기는 416x416크기의 해상도를 사용 합니다. 그 이유는 최족 특징맵 크기를 홀수x홀수로 만들기 위함에 있는데 그리드 
셀이 특징맵 중앙을 검출하는데 효과가 있습니다. 특히 큰 오브젝트라면 중앙을 차지하는 경향이 있기에 근처의 위치보다 단일한 중앙의 
위치를 찾는 것이 효과적 입니다. 최종적으로는 YoloV2에서 32배만큼 축소가 이루어지므로 416x416의 이미지를 사용하면 13x13의 
feature map을 얻을 수 있게 됩니다.

## Anchor Box 

기존 Yolo에서는 2개의 경계 박스를 제안하고 경계 박스들은 Class확률을 공유하였습니다.
V2에서는 각각의 5개의 Anchor Box들이 존재하고 Anchor Box들은 독립적으로 클래스 확률을 가지고 있습니다.
즉, 최종적으로 클래스의 예측은 예측한 경계박스가 오브젝트일때 그것이 어떤 클래스인지 예측하는 조건부 확률이 됩니다.

경계박스를 사용하기 전은 69.5 mAP, recall 81%
경계박스를 사용한 후는 69.2 mAP, recall 88%로 
mAP는 조금 떨어졌지만 recall은 크게 상승했습니다.

## Demension Clusters

YoloV2의 Anchor box들의 종횡비는 Define 되어서 사용하는데 Ahchor box의 종횡비는 VOC나 COCO 데이터셋 등의
실제 오브젝트의 경계박스들을 k-mean Cluster를 통해 클러스터링 하였습니다.
여기서 유클라디안 거리를 이용한 k-mean을 사용하면 경계박스가 커질수록 에러율이 커지게 됩니다.
그렇기 때문에 IOU점수를 높여 좋은 Anchor box를 선택하는 거리 공식을 사용하였습니다. 공식은 다음과 같습니다.
  
  * d(box, centroid) = 1 − IOU(box, centroid)

k와 avg IOU의 관계를 그래프

![anchor cluster](https://user-images.githubusercontent.com/44501825/49169747-ded89e80-f37d-11e8-904e-134960ed5562.jpg)

왼쪽 그래프를 보면 k-mean cluster
의 클러스터링 갯수인 k를 크게 할 수록 IOU가 커져 recall이 상승하게 됩니다. 
하지만 recall과 model의 복잡성을 고려하여 k=5로 사용합니다.
## Direct location prediction

* Yolo 와 Anchor box를 같이 사용시 발생하는 문제
모델이 학습 초기에 불안정성을 보여주는 문제가 있습니다. 그 원인으로는 경계박스(x,y)위치를 예측하는
부분에서 문제가 발생합니다.
지역 제안 네트워크(RPN)에서 tx와 ty 그리고 중심좌표(x,y)를 다음과 같이 구합니다.

![rpn](https://user-images.githubusercontent.com/44501825/49885852-a2c43400-fe7b-11e8-91d9-fa4f6e257a41.jpg)

예를 들어 설명하면 tx = 1이면 앵커박스는 폭만큼 오른쪽, tx = -1 이면 앵커박스는 폭만큼 왼쪽으로 이동시킵니다.
이 공식을 사용했을때 문제점은 제약이 없다는 것입니다. 영상의 어느 위치에 경계박스를 예측하는가 상관없이 이미지
전체에서 어디에도 나타날 수 있게 됩니다. 그렇기 때문에 무작위 초기화를 통해 오프셋을 예측하는것은 안정화되기 까지
오랜 시간이 소요됩니다.

그래서 Yolo팀은 오프셋은 예측하는 대신 그리드 셀 위치를 기준으로 위치좌표(x,y)를 예측합니다.
x, y의 GT(Ground Truth)은 항상 0 ~ 1 사이의 값이 되며 예측할 sigmoid 함수를 사용하여 범위가 0 ~ 1 이 되도록 하였습니다.

네트워크가 출력하는 feature map의 각 셀에 5개의 경계박스를 예측합니다. (클러스터링을 통해 찾은 5개의 Anchor box)
즉, tx, ty, tw, th 그리고 to에 대해서 5개의 좌표를 예측합니다.

예측 공식은 다음과 같습니다.

![detection local prediction](https://user-images.githubusercontent.com/44501825/49888747-21709f80-fe83-11e8-915d-753184f5eec5.jpg)

(cx, cy)는 이미지 좌상단으로 부터의 offset,  경계 상자의 폭과 높이는 pw, ph 입니다.


![location prediction](https://user-images.githubusercontent.com/44501825/49957581-a1fad300-ff4b-11e8-9ff4-fb8202435b04.jpg)

tx, ty는 시그모이드 함수를 거치게 되는데 시그모이드 함수를 거친 결과가 0.5라는 값을 기대합니다.
즉, tx와 ty는 0이 되도록 유도하는 것이며 시그모이드를 거친 tx, ty값이 0.5라는 것은 그리드셀의 중심 좌표를 의미합니다.

## Fine-Grained Features

YoloV2에서는 13x13의 Feature map에서 오브젝트를 예측하는데 이 방법은 큰 오브젝트를 탐지하는 것에는 적합할 수 있으나
작은 오브젝트를 탐지하는 것에는 충분하지 못합니다.

그렇기 때문에 Yolo팀은 26x26의 해상도의 이전계층 Layer를 스킵하여 13x13 Layer에 연결하는 Passthrough Layer를 추가하였습니다.

이 방법으로 Yolo는 1%의 성능향상이 있었습니다. 


## 학습결과



참고논문 : YOLO9000:Better, Faster, Stronger 

Link : https://pjreddie.com/darknet/yolo/
  

