원본 : https://github.com/golbin/TensorFlow-Tutorials/tree/master/11%20-%20Inception

# Inception 모델 사용해보기

### 학습시켜 볼 샘플 자료 다운로드

```
http://download.tensorflow.org/example_images/flower_photos.tgz
```

자신이 가진 다른 이미지를 학습시켜보고 싶다면, 학습시킬 사진을 각각의 레이블 이름으로 폴더를 생성하고, 그 안에 폴더 이름에 맞는 사진을 넣어두면 됩니다.

### 학습 실행

```
# python retrain.py     --bottleneck_dir=./workspace/bottlenecks     --model_dir=./workspace/inception     --output_graph=./workspace/flowers_graph.pb     --output_labels=./workspace/flowers_labels.txt     --image_dir ./workspace/flower_photos     --how_many_training_steps 1000
```

### retrain.py 주요 옵션

- --bottleneck_dir : 학습할 사진을 인셉션 용으로 변환해서 저장할 폴더
- --model_dir : inception 모델을 다운로드 할 경로
- --image_dir : 원본 이미지 경로
- --output_graph : 추론에 사용할 학습된 파일(.pb) 경로
- --output_labels : 추론에 사용할 레이블 파일 경로
- --how_many_training_steps : 얼만큼 반복 학습시킬 것인지

### 추론 테스트

```
# python predict.py ./workspace/flower_photos/roses/20409866779_ac473f55e0_m.jpg
```
