## 估算不同样本之间的相似性度量(Similarity Measurement)

#### 1. Euclidean Distance(欧氏距离)
- $d_{12} = \sqrt{(x_1-x_2)^2+(y_1-y_2)^2}$
#### 2. Manhattan/City Block Distance(曼哈顿距离)
- $d_{12} = |x_1-x_2|+|y_1-y_2|$
#### 3. Chebyshev Distance(切比雪夫距离)
- $d_{12} = max(|x_1-x_2|,|y_1-y_2|)$
#### 4. Minkowski Distance(闵可夫斯基距离)
- $d_{12} = \sqrt[p]{\sum^n_{k=1}|x_{1k}-x_{2k}|^p}$
- p=1,Manhattan;p=2,Euclidean;p=$\infty$,Chebyshev; 
#### 5. Standardized Euclidean distance(标准化欧氏距离)
- $d_{12} = \sqrt{\sum^n_{k=1}(\frac{x_{1k}-x_{2k}}{S_k})^2}$
#### 6. Mahalanobis Distance(马氏距离)
- $d(x) = \sqrt{(x-\mu)^TS^{-1}(x-\mu)} $
- $d(x_i,x_j) = \sqrt{(x_i-x_j)^TS^{-1}(x_i-x{_j})}$
- S为协方差矩阵，$\mu$为均值
#### 7. Cosine(夹角余弦)
- $cos\theta = \frac{x_1x_2+y_1y_2}{\sqrt{x_1^2+y_1^2}\sqrt{x_2^2+y_2^2}}$
- 余弦取值范围为[-1,1]
#### 8. Hamming distance(汉明距离)
- 两个等长字符串s1与s2之间的汉明距离定义为将其中一个变为另外一个所需要作的最小替换次数
#### 9. Jaccard similarity coefficient(杰卡德距离 & 杰卡德相似系数)
- $J(A,B) = \frac{|A \cap B|}{|A \cup B|}$
- $J_{\sigma}(A,B) = 1-J(A,B)$
- 杰卡德距离用两个集合中不同元素占所有元素的比例来衡量两个集合的区分度,用在衡量样本的相似度上
#### 10. Pelson Correlation coefficient(皮尔逊相关系数 & 相关距离)
- $\rho_{xy}= \frac{Cov(x,y)}{\sqrt{D(x)}\sqrt{D(y)}}$
- $ Cov(x,y) = E((x-Ex)(y-Ey))$
- $D_{xy} = 1-\rho_{xy}$
#### 11. Information Entropy信息熵
- $Entropy(x) =\sum^n_{i=1}-P_ilog_2P_i$
