
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
여기서 YoloV2와 차이점은 기존 Yolo는 마지막 단에 FC단에서 경계박스를 직접 예측하였지만 V2에서는 FC를 Convloution으로 대체하고
Anchor Box를 이용해 경계 박스를 예측합니다.
그리고 입력 이미지크기는 416x416크기의 해상도를 사용 합니다. 그 이유는 최족 특징맵 크기를 홀수x홀수로 만들기 위함에 있는데 그리드 
셀이 특징맵 중앙을 검출하는데 효과가 있습니다. 특히 큰 오브젝트라면 중앙을 차지하는 경향이 있기에 근처의 위치보다 단일한 중앙의 
위치를 찾는 것이 효과적 입니다. 최종적으로는 YoloV2에서 32배만큼 축소가 이루어지므로 416x416의 이미지를 사용하면 13x13의 
feature map을 얻을 수 있게 됩니다.

## Anchor Box 

기존 Yolo에서는 2개의 경계 박스를 제안하고 경계 박스들은 Class확률을 공유하였습닏.
V2에서는 각각의 5개의 Anchor Box들이 존재하고 Anchor Box들은 독립적으로 클래스 확률을 가지고 있습니다.
즉, 최종적으로 클래스의 예측은 예측한 경계박스가 오브젝트일때 그것이 어떤 클래스인지 예측하는 조건부 확률이 됩니다.

경계박스를 사용하기 전은 69.5 mAP, recall 81%
경계박스를 사용한 후는 69.2 mAP, recall 88%로 
mAP는 조금 떨어졌지만 recall은 크게 상승했습니다.

## Demension Clusters

YoloV2의 Anchor box들의 종횡비는 Define 되어서 사용하는데 Ahchor box의 종횡비는 VOC나 COCO 데이터셋 등의
실제 오브젝트의 경계박스들을 k-mean Cluster를 통해 클러스터링 하였습니다.
적절한 Anchor box를 사용하는 것은 IOU점수를 높히는 방법이기에 이 작업은 중요하다고 볼 수 있습니다.

![anchor cluster](https://user-images.githubusercontent.com/44501825/49169747-ded89e80-f37d-11e8-904e-134960ed5562.jpg)

왼쪽 그래프를 보면 k-mean cluster의 클러스터링 갯수인 k를 크게 할 수록 IOU가 커져 recall이 상승 하게됩니다. 
하지만 recall과 model의 복잡성을 고려하여 k=5로 하여 사용합니다.

## 학습결과




참고논문 : YOLO9000:Better, Faster, Stronger 

Link : https://pjreddie.com/darknet/yolo/
  

