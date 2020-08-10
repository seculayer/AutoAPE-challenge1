데이터 구성
-배가 한 이미지에 여러 개 등장하기도 하지만 배가 아예 등장 안하는 사진도 많음--> 반드시 eda과정에서 처리 방향 결정
->brightness 고려하여 data를 augment하기(by ImageDataGenerator)


-train_ship_segmentations.csv file: training image에 대한 ground truth 제공
평가지표
 F2 Score at different intersection over union (IoU) thresholds
 
 
 
 코드 짠 원리
 -extract segmentation map for ship
 -augment image and train simple DNN model to detect them. 
 
 
model은 U-Net standard에서 약간 변형을 줌 

improve Deep Learning Model Robustness by Adding Noise


헷갈리는 부분: decode RLE into Image
