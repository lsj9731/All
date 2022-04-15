https://github.com/BorgwardtLab/Set_Functions_for_Time_Series 논문을 구현

PhysioNet2012 시퀀스 데이터를 활용하여 Transformer & Attention을 통해 병원 내 사망을 예측

using tensorflow 2.x version

모델 성능 평가 지표 - Accuracy, AUC score
-------------------------------------------

Paper Acc : 83.7 ± 3.5 AUC score : 86.3 ± 0.8

Implemented Acc : 79.8 AUC score : 83.4

-------------------------------------------
Get test result -> python test.py --checkpoint early_stop/14epochs/Model_Best_Checkpoint_14_epochs
