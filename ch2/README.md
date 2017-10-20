# 고양이 강아지 분류하기

##  데이터 준비
https://www.kaggle.com/c/dogs-vs-cats/data <br/>
회원가입 후 train.zip 다운 받기 <br/>
압축 풀어서 현재 경로에 넣는다  <br />
convet, test 폴더를 만든다<br />

## 이미지 크기 조절
python convert_make.py
이미지가 너무 크고 다 다르기 때문에 <br/>
128 * 128 이미지로 변환한다. <br/>
python conver_make.py 숫자 입력시 train set 크기를 늘릴 수 있다 <br/>
기본 실행 시 train set : 3000, test set : 1000 로 되어 있다<br/>
python conver_make.py {숫자} 입력시 train set 크기를 늘릴 수 있다<br/>
최대 25,000 까지 가능<br/>
데이터 셋 더 많이 만들고 싶으면 직접 이미지를 넣거나 이미지 분할을 통해서 데이터를 늘린다<br/>

## 기본 코드 실행
python base.py
코드 마지막에 test 데이터를 테스트하면서 실제로 볼 수 있다

## 목표
base.py 실행 시 정확도가 50% 나온다 <br/>
네트워크 레이어 수정, 함수 수정, 데이터 셋 더 늘리기 등 여러 방법으로 정확도를 높여보자 <br/>
