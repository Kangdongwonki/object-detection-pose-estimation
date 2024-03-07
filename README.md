# 사전 세팅
- ubuntu 20.04
- conda 환경 설치 python=3.7
- realsense2-ros 설치
- 학습된 모델 파일


# 가상환경 세팅
- opencv-python=4.8.1
- pandas=1.3.5
- scikit-image=0.19.3
- scikit-learn=1.0.2
- scipy=1.4.1
- seaborn=0.12.2
- tensorflow-gpu=2.3.0


# 패키지 생성
- work space를 생성하여 src 폴더에 파일들을 첨부하고 catkin-make 수행
- 스크립트 파일은 ws/src/script
- 모델 파일은 ws/src


# 실행
각각의 터미널을 열어서
1. roscore 실행
2. roslaunch realsense2_camera rs_camera.launch 실행
3. rosrun [패키지 이름] [노드 실행 파일 이름]
4. rostopic echo [토픽 이름]


# 설명
기존의 공개 데이터 셋 COCO 모델을 파인 튜닝하여 RGBD 카메라를 통해 객체 감지와 감지된 객체와의 관계식을 통해서 객체의 포즈 추정 수행

<img width="426" alt="roscam" src="https://github.com/Kangdongwonki/object-detection-pose-estimation/assets/94422945/a9cece4b-7d25-4ade-9530-2c1992cfd3e7">
