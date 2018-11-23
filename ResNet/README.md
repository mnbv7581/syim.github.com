
# syim.github.com


cifar 데이터셋을 활용하여 만든 ResNet Layer 입니다
해당 소스는 python Tensorflow기반으로 만들었습니다.

#  Network Architectures

![resnet-layer](https://user-images.githubusercontent.com/44501825/48914301-5d38ca80-eebe-11e8-924b-041a84814591.jpg)

왼쪽 - VggNet
가운데 - PlainNet
오른쪽 - Resnet

기존 Vggnet의 개념을 참고하여 Resnet팀은 PlainNet을 구현했다고 합니다. 
즉, Convolution들은 대게 3x3 필터를 가지고 있고 두 가지 규칙을 가집니다.
1. 동일한 출력 피쳐 맵 크기의 경우 레이어의 필터 수가 동일
2. 피쳐맵의 사이즈가 절반 인 경우 필터의 수는 2배가 되며 Stride가 2인 Convolution Layer에 의해 
직접 다운 샘플링 합니다(VGG의 경우 Max Pooling을 사용)


![plainnet test error](https://user-images.githubusercontent.com/44501825/48915563-75aae400-eec2-11e8-9722-9fc526f656a4.jpg)

하지만 PlainNet의 Layer층의 깊이가 깊어질수록 훈련오류가 커지는 문제가 발생 하였습니다.
그림에서의 그래프를 확인해보면 20-Layer보다 56-Layer가 에러율이 더 큰 것을 확인되었습니다.
Resnet팀은 문제를 개선하기위해 "shortcut connections"이라는 개념을 도입하여 에러율을 줄였습니다.

## shortcut connections

![shortcut connections](https://user-images.githubusercontent.com/44501825/48916734-c9b7c780-eec6-11e8-8e95-34b7de2822ed.jpg)

Layer가 깊으면 깊을수록 학습이 잘 된다는 것이 딥러닝에서의 일반적인 생각이지만 실제 결과적으로는 그렇지 않았습니다.
Layer가 깊어질수록 Gradient의 전달문제가 발생하여 이 문제를 해결하기위해 [BN -> Relu -> Conv] 과정을 거친 F(x)에
이과정을 거치기 바로 전인 x를 더하여 Gradient 전달 문제를 해결하였습니다.


# 학습결과




참고논문 : Deep Residual Learning for Image Recognition[Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun]Microsoft Research {kahe, v-xiangz, v-shren, jiansun}@microsoft.com

  
