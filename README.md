# Image02
カメラ画像を取得して処理するコード

camera02.pyの説明

opencvを用いてカメラで取得した画像に次の6つの画像処理を行った。

1. ガンマ変換によるコントラスト変換

2. BGR色調の各色の調整

3. HSV色調にカラーを変換

4. 平滑化フィルタ

5. ラプラシアンフィルタ

6. 先鋭化でエッジの強調


使い方

実行する一つの処理に対してのみ、スイッチをON(1)にすることで処理をする。

1,2,4,5はトラックバーを」操作してパラメータを変えられる。

終了はqを入力。

依存ライブラリとバージョン

from scipy.ndimage.filters import convolve や from numpy import sqrt

Opencvバージョン:'3.4.1'

参考文献1(スイッチのコードで参考)

https://sites.google.com/site/lifeslash7830/home/hua-xiang-chu-li/opencvniyoruhuaxiangchulitorakkubatoka?tmpl=%2Fsystem%2Fapp%2Ftemplates%2Fprint%2F&showPrintDialog=1

参考文献2(HSVへの変換式を参考)

http://lang.sist.chukyo-u.ac.jp/classes/OpenCV/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html


参考文献3(ラプラシアンフィルタの引用)

https://qiita.com/shim0mura/items/3ab2a67c78eafd456c32



実行の様子

youtube: https://youtu.be/6y5Iq4JzP9k

