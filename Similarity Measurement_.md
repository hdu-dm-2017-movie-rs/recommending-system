## 估算不同样本之间的相似性度量(Similarity Measurement)
  
  
#### 1. Euclidean Distance(欧氏距离)
  
- <img src="https://latex.codecogs.com/gif.latex?d_{12}%20=%20&#x5C;sqrt{(x_1-x_2)^2+(y_1-y_2)^2}"/>
#### 2. Manhattan/City Block Distance(曼哈顿距离)
  
- <img src="https://latex.codecogs.com/gif.latex?d_{12}%20=%20|x_1-x_2|+|y_1-y_2|"/>
#### 3. Chebyshev Distance(切比雪夫距离)
  
- <img src="https://latex.codecogs.com/gif.latex?d_{12}%20=%20max(|x_1-x_2|,|y_1-y_2|)"/>
#### 4. Minkowski Distance(闵可夫斯基距离)
  
- <img src="https://latex.codecogs.com/gif.latex?d_{12}%20=%20&#x5C;sqrt[p]{&#x5C;sum^n_{k=1}|x_{1k}-x_{2k}|^p}"/>
- p=1,Manhattan;p=2,Euclidean;p=<img src="https://latex.codecogs.com/gif.latex?&#x5C;infty"/>,Chebyshev; 
#### 5. Standardized Euclidean distance(标准化欧氏距离)
  
- <img src="https://latex.codecogs.com/gif.latex?d_{12}%20=%20&#x5C;sqrt{&#x5C;sum^n_{k=1}(&#x5C;frac{x_{1k}-x_{2k}}{S_k})^2}"/>
#### 6. Mahalanobis Distance(马氏距离)
  
- <img src="https://latex.codecogs.com/gif.latex?d(x)%20=%20&#x5C;sqrt{(x-&#x5C;mu)^TS^{-1}(x-&#x5C;mu)}"/>
- <img src="https://latex.codecogs.com/gif.latex?d(x_i,x_j)%20=%20&#x5C;sqrt{(x_i-x_j)^TS^{-1}(x_i-x{_j})}"/>
- S为协方差矩阵，<img src="https://latex.codecogs.com/gif.latex?&#x5C;mu"/>为均值
#### 7. Cosine(夹角余弦)
  
- <img src="https://latex.codecogs.com/gif.latex?cos&#x5C;theta%20=%20&#x5C;frac{x_1x_2+y_1y_2}{&#x5C;sqrt{x_1^2+y_1^2}&#x5C;sqrt{x_2^2+y_2^2}}"/>
- 余弦取值范围为[-1,1]
#### 8. Hamming distance(汉明距离)
  
- 两个等长字符串s1与s2之间的汉明距离定义为将其中一个变为另外一个所需要作的最小替换次数
#### 9. Jaccard similarity coefficient(杰卡德距离 & 杰卡德相似系数)
  
- <img src="https://latex.codecogs.com/gif.latex?J(A,B)%20=%20&#x5C;frac{|A%20&#x5C;cap%20B|}{|A%20&#x5C;cup%20B|}"/>
- <img src="https://latex.codecogs.com/gif.latex?J_{&#x5C;sigma}(A,B)%20=%201-J(A,B)"/>
- 杰卡德距离用两个集合中不同元素占所有元素的比例来衡量两个集合的区分度,用在衡量样本的相似度上
#### 10. Pelson Correlation coefficient(皮尔逊相关系数 & 相关距离)
  
- <img src="https://latex.codecogs.com/gif.latex?&#x5C;rho_{xy}=%20&#x5C;frac{Cov(x,y)}{&#x5C;sqrt{D(x)}&#x5C;sqrt{D(y)}}"/>
- <img src="https://latex.codecogs.com/gif.latex?Cov(x,y)%20=%20E((x-Ex)(y-Ey))"/>
- <img src="https://latex.codecogs.com/gif.latex?D_{xy}%20=%201-&#x5C;rho_{xy}"/>
#### 11. Information Entropy信息熵
  
- <img src="https://latex.codecogs.com/gif.latex?Entropy(x)%20=&#x5C;sum^n_{i=1}-P_ilog_2P_i"/>
  