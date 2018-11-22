
# syim.github.com


cifar 데이터셋을 활용하여 만든 VGG-16 Layer 입니다
해당 소스는 python Tensorflow기반으로 만들었습니다.

#  Network Architectures

![vgg-layer table](https://user-images.githubusercontent.com/44501825/48899188-3cf31680-ee92-11e8-9a4d-228d12ccab8c.jpg)


Input->Convolution1->Batch Normalization->Relu->Max Pooling...
    Convolution5->Batch Normalization->Relu->Max Pooling->fully Connected


## 1.Convolution

![vgg abstract](https://user-images.githubusercontent.com/44501825/48889093-090af780-ee78-11e8-8fbb-08e5b86f68b8.jpg)

논문(Very Deep Convolutional Networks for Large-Scale Image RecognitionKaren Simonyan, Andrew Zisserman) 내용을 살펴보면 VGG Layer에서는 3x3의 작은 Filter크기를 이용해 깊게(16-19) 가중치 레이어를 만들면 좋은 결과를 얻었다는 내용이 있습니다.
이게 따라 소스구현시 3x3의 필더크기로 convolution layer를 구성하였습니다.


## 2.Batch Normalization

![batchnormal](https://user-images.githubusercontent.com/44501825/48889303-b0882a00-ee78-11e8-97b1-363cfd207809.png)

뉴럴넷 학습시 보통 mini-batch 단위로 데이터를 가져와서 학습을 시키는데 각 feature 별로 평균과 표준편차를 구해준다음 Normalize 해줍니다.

즉, 평균은0, 표준편차는 1인 분포를 만드는 과정이라고 생각하면 된다(좀더 디테일게 들어가면 다른 변수들이 등장하긴 함)
논문 내용을 보면 분산을 구한 다음 분산에 epsilon이라는 미세한 값을 더 한것의 제곱근으로 나누는데 결국 표준편차를 나누는 것이라고
생각하면 된다. 그렇게 [표준값, 표준점수]를 구하게 됨(x의 값이 평균에서 얼만큼 떨어져 있는지를 계산하게 됨)
이렇게 각 레이어의 분포를 같게 함으로써 안정적이게 학습할 수 있게 해줍니다.

## 3.Relu(Activation function)
![sigmoid-relu](https://user-images.githubusercontent.com/44501825/48898424-146a1d00-ee90-11e8-83b4-ef8c4b353377.png)


Sigmoid Function을 Relu로 대체하게 되면 Gradient Vanishing문제를 개선 할 수 있다고 합니다.
Sigmoid Fucntion의 경우 역전파 수행시 Layer를 지날수록 Gradient값을 곱해 결국 0의 수렴하는 값이 되버립니다(함수값이 0~1 사이에만 존재하기 때문)
그와 다르게 Relu Function은 0보다 작은 수는 0이고 0보다 크면 입력값 그대로 내보낸다.

Relu 장점 
1. 0이하의 입력에대해 0을 출력함으로써 부분적으로 활성화 시킬수 있다
2. Gradient Vanishing이 없으며 Gradient가 증가하지 않는다
3. 선형함수이므로 미분 계산이 간단하다.

## 4. Max pooling 

![max pooling](https://user-images.githubusercontent.com/44501825/48899305-99563600-ee92-11e8-89fc-f5967921559b.png)

여러가지 pooling 기법중 많이 사용되는 Max Pooling을 사용하여 Sampling 과정을 거쳤습다.
특정 Feature가 존재할 때 가장 큰 Activation값을 가지게하여 효과적일 수 있다고며 실제로 다른 Pooling 기법보다
더 좋은 결과를 보여줬다고 합니다.(2x2 filter, stride=2 사용함)

## 5. Fully Connected

<img width="500" alt="fully connected" src="https://user-images.githubusercontent.com/44501825/48899953-64e37980-ee94-11e8-8482-decfa8a9b3b3.png">

최종적으로 Fully Connected 과정을 통해 최종 Layer를 Class들을 분류한다.
즉, 최종 Layer를 1차원으로 길에 늘린뒤 가중치들을 최종 Fully connected Layer 하나하나에 연결시켜주는 과정을
거쳐준다 
이과정에서 가중치들을 Dropout 시켜주어 overfitting을 예방한다.(학습시 가중치에 너무 과적합되어 버릴수 있어 일부 가중치를 랜덤하게 버려준다)


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




  
