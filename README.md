## camera-calibration-evaluation

评估手眼标定精度，计算平移xyz与旋转的RMSE

[手眼标定部分更改自](https://git.lug.ustc.edu.cn/GWDx/camera-calibration)

[评估部分更改自](https://github.com/ethz-asl/hand_eye_calibration)

### 使用方法：
图片放在 data/chessboard/ 中，位姿文件放在 data/chessboard/pose.xml 中

`conda install --yes --file requirements.txt`

`python calibration.py`
