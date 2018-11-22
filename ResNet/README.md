
# syim.github.com


cifar 데이터셋을 활용하여 만든 ResNet Layer 입니다
해당 소스는 python Tensorflow기반으로 만들었습니다.

#  Network Architectures

![resnet-layer](https://user-images.githubusercontent.com/44501825/48914301-5d38ca80-eebe-11e8-924b-041a84814591.jpg)

왼쪽 - VggNet
가운데 - PlainNet
오른쪽 - Resnet

기존 Vggnet의 개념을 참고하여 Resnet팀은 PlainNet을 구현했다고 한다. 
즉, Convolution들은 대게 3x3 필터를 가지고 있고 두 가지 규칙을 가집니다.
1. 동일한 출력 피쳐 맵 크기의 경우 레이어의 필터 수가 동일
2. 피쳐맵의 사이즈가 절반 인 경우 필터의 수는 2배가 되며 Stride가 2인 Convolution Layer에 의해 
직접 다운 샘플링 합니다(VGG의 경우 Max Pooling을 사용)


## 1.Convolution

![vgg abstract](https://user-images.githubusercontent.com/44501825/48889093-090af780-ee78-11e8-8fbb-08e5b86f68b8.jpg)

논문(Very Deep Convolutional Networks for Large-Scale Image RecognitionKaren Simonyan, Andrew Zisserman) 내용을 살펴보면 VGG Layer에서는 3x3의 작은 Filter크기를 이용해 깊게(16-19) 가중치 레이어를 만들면 좋은 결과를 얻었다는 내용이 있습니다.
이게 따라 소스구현시 3x3의 필더크기로 convolution layer를 구성하였습니다.


# 학습결과

50000장의 트레이닝 셋과 10000장의 테스트 셋에 대한 결과이다.

최종적으로 test 77% 정도의 정확도를 보여주었다

epoch : 206, loss : 0.004638576596625251, accuracy : 1.0

epoch : 207, loss : 0.007005987549474279, accuracy : 0.9999554843304843

epoch : 208, loss : 0.009715790177669003, accuracy : 0.9998219373219374

epoch : 209, loss : 0.004735732453029823, accuracy : 0.9998441951566952

epoch : 210, loss : 0.005462177022871614, accuracy : 0.9998441951566952

epoch 210 valid set accuracy : 0.7782451923076923

final test set accuracy : 0.7724358974358975




  
