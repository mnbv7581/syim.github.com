
# syim.github.com


PASCAL VOC(Visual Object Classes Challenge) 데이터셋 기반으로 만든 YoloV2 입니다. CNN layer는 Resnet을 사용하였습니다.




##  Network ArchitecturesY

*YoloV1

![yolov1](https://user-images.githubusercontent.com/44501825/49067351-2ddfe000-f267-11e8-8ce8-1e712c06e346.jpg)


YoloV1의 Convolution Layer 구성을 보면 448x448x3의 Input 이미지가 최종적으로는 FC를 통해 7x7x30의 output을 생성합니다.
여기서 YoloV2와 차이점은 V1는 마지막 단에 FC단에서 경계박스를 직접 예측하였지만 V2에서는 FC를 Convloution으로 대체하여
Anchor Box를 이용하여 Bounding box를 예측합니다.
입력 이미지크기는 416x416크기의 해상도를 사용 합니다. 그 이유는 최족 특징맵 크기를 홀수x홀수로 만들기 위함에 있는데 그리드 
셀이 특징맵 중앙을 검출하는데 효과가 있다. 특히 큰 오브젝트라면 중앙을 차지하는 경향이 있기에 근처의 위치보다 단일한 중앙의 
위치를 찾는 것이 효과적 입니다. 최종적으로는 YoloV2에서 32배만큼 축소가 이루어지므로 416x416의 이미지를 사용하면 13x13의 
feature map을 얻을 수 있게 됩니다.




## 학습결과




참고논문 : YOLO9000:Better, Faster, Stronger 

Link : https://pjreddie.com/darknet/yolo/
  

