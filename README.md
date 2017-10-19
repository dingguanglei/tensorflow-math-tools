<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?c </script>

# tensorflow-math-tools
常用函数集合
## 评估函数类

### MAE :Mean Absolute Error （平均绝对误差）
    平均绝对误差是绝对误差的平均值
    平均绝对误差能更好地反映预测值误差的实际情况。

```math
MAE={\frac{1}{N}\sum_{i=1}^{N}\lvert(f_i-y_i)\rvert}
```


```python
import my_tensorflow_operation as mytf
y1 = np.array([1., 2., 3., 4., 5., 6., 7.])
y3 = np.array([2., 4., 6., 8., 10., 12., 14.])
loss=mytf.reduce_MAE(y1,y3)
sess=tf.Session()
print(sess.run(loss))#==> 4.0
```
### MAPE :Mean Absolute Error （平均绝对误差）
    平均绝对误差是绝对误差的平均值
    平均绝对误差能更好地反映预测值误差的实际情况。

```math
MAPE={\frac{1}{N}\sum_{i=1}^{N}\lvert(\frac{f_i-y_i}{f_{i}})\rvert}
```


```python
import my_tensorflow_operation as mytf
y1 = np.array([1., 2., 3., 4., 5., 6., 7.])
y2 = np.array([1.2, 2.2, 3.2, 4.2, 5.1, 6.3, 7.2])
y3 = np.array([2., 4., 6., 8., 10., 12., 14.])
# y1 must be real data. y2 and y3 must be predict data
loss12=mytf.reduce_MAPE(y1,y2)
loss13=mytf.reduce_MAPE(y1,y3)
sess=tf.Session()
print(sess.run(loss12))#==> 0.0736054421769
print(sess.run(loss13))#==> 1.0
```
### Pearson correlation coefficient(皮尔森相关系数)
    也称皮尔森积矩相关系数，是一种线性相关系数 。
    皮尔森相关系数是用来反映两个变量线性相关程度的统计量。
    相关系数用r表示，其中n为样本量，分别为两个变量的观测值和均值。
    r描述的是两个变量间线性相关强弱的程度。
    r的绝对值越大表明相关性越强。

```math
Person_{x,y}={\frac{cov(X,Y)}{\delta_{X}\delta_{Y}}\ }={\frac{E((X-\mu_{X})(Y-\mu{Y}))}{\delta_{X}\delta_{Y}} }
```
化简后：
```math
Person_{x,y}={\frac{\sum{XY}-\frac{\sum{X}\sum{Y}}{N} }{\sqrt{(\sum{X^2}-\frac{(\sum{X})^2}{N} }\sqrt{(\sum{Y^2}-\frac{(\sum{Y})^2}{N} } }}
```

```python
import my_tensorflow_operation as mytf
y1 = np.array([1., 2., 3., 4., 5., 6., 7.])
y2 = np.array([1.2, 2.2, 3.2, 4.2, 5.1, 6.3, 7.2])
y3 = np.array([2., 4., 6., 8., 10., 12., 14.])
person12=mytf.reduce_MAPE(y1,y2)
person13=mytf.reduce_MAPE(y1,y3)
sess=tf.Session()
print(sess.run(person12))  #==>0.999651908638
print(sess.run(person13))  #==>1.0
```
