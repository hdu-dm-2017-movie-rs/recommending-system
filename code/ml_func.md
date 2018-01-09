## 推荐系统算法部分总结
### 一、使用数据集
- *1.[MovieLens](http://www.grouplens.org/node/73)
*2.豆瓣
3.Netflix
4.IMDb
- 最终选择movielens的ml-latest作为训练数据集，爬取豆瓣用户数据作为测试和推荐的数据集
---
## 二、传统算法研究
#### 协同过滤算法
  - 使用`用户-物品`，`物品-评分`格式的数据
  - 假设有用户u和v，物品i和j，采用Jaccard公式或cos相似度的计算
	- 1.基于用户(UserCF)：评测用户之间的相似性，给用户推荐和他兴趣相投的`其他用户`喜欢的物品
		- 构建**用户**相似度矩阵w
		- **相似度计算**：$W_{uv} = \frac{|N(u)\cap N(v)|}{\sqrt{|N(u)||N(v)|}}$
			- N(u)表示用户喜欢的物品的集合
		- **相似度计算改进1**：首先计算出$|N(u)\cap N(v)|\neq 0$的用户对$(u,v)$进行初步筛选,之后用余弦相似度计算
		- **相似度计算改进2**增加量惩罚机制，惩罚用户u和用户v共同兴趣列表中热门物品对他们相似度的影响
      - $W_{uv} = \frac{|N(u)\cap N(v)|\frac1{log1+N(i)}}{\sqrt{|N(u)||N(v)|}}$

	  - **推荐公式**：$P(u,i) = \sum_{v\in S(u.k)\cap N(i)}W_{uv}R_{vi}$
			- S(u,k)表示和用户u兴趣最接近的K个用户，N(i)表示对物品i有过行为的用户集合，$W_uv$表示用户u和用户v的兴趣相似度，$R_vi$表示用户v对物品i的评分)
			- 选取不同k进行评测，得出最佳推荐数量的物品
	
		```python
		# 相似度计算
		def UserSimilarity(train):
			W = dict()
			for u in train.keys():
				for v in train.keys():
					if u == v:
						continue
					W[u][v] = len(train[u] & train[v])
					W[u][v] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)
			return W

		# 推荐
		def Recommend(user, train, W):
			rank = dict()
			interacted_items = train[user]
			for v, wuv in sorted(W[u].items, key=itemgetter(1), \
				reverse=True)[0:K]:
			for i, rvi in train[v].items:
				if i in interacted_items:
					#we should filter items user interacted before
					continue
				rank[i] += wuv * rvi
			return rank
		```	

	- 2.基于物品(ItemCF)：评测物品之间的相似性，给用户推荐和他之前喜欢物品`相似的物品`
		- 构建**物品**相似度矩阵w
		- **相似度计算1**：$W_{ij} = \frac{|N(i)\cap N(j)|}{|N(i)|}$
			- N(i)表示对物品i有过行为的用户集合，可以理解为喜欢物品i的用户中有多少比例的用户也喜欢物品j
		- **相似度计算2**$W_{ij} = \frac{|N(i)\cap N(j)|}{\sqrt{|N(i)||N(j)|}}$
			- 如果物品j很热门，很多人都喜欢，那么$W_ij$就会很大,需要避免一直推荐出太热门的物品
		- **相似度计算改进**：软性惩罚机制 
    $W_{ij} = \frac{|N(i)\cap N(j)|\frac1{log1+|N(u)|}}{\sqrt{|N(i)||N(j)|}}$
			$W_{ij} = \frac{|N(i)\cap N(j)|}{|N(i)|^{1-a}|N(j)|^a} a\in[0.5,1]$
		- 推荐公式：$P(i,j) = \sum_{v\in S(j.k)\cap N(u)}W_{ij}R_{ui}$
			- S(j,K)是和物品j最相似的K个物品的集合，N(u)表示用户喜欢的物品的集合，$W_{ji}$是物品j对i的相似度，$R_{ui}$是用户u对物品i的兴趣
		- **归一化**：用于提升准确度，覆盖率和多样性 $W_ij` = \frac{W_ij}{max_j W_ij}$
  
		```python
		# 相似度计算
		def ItemSimilarity(train):
			#calculate co-rated users between items
			C = dict()
			N = dict()
			for u, items in train.items():
				for i in users:
					N[i] += 1
					for j in users:
						if i == j:
							continue
						C[i][j] += 1
						# C[i][j] += 1 / math.log(1+len(items)*1.0)
			
			#calculate finial similarity matrix W
			W = dict()
			for i,related_items in C.items():
				for j, cij in related_items.items():
					W[u][v] = cij / math.sqrt(N[i] * N[j])
			return W

		# 推荐公式
		def Recommendation(train, user_id, W, K):
			rank = dict()
			ru = train[user_id]
			for i,pi in ru.items():
				for j, wj in sorted(W[i].items(),key=itemgetter(1), reverse=True)[0:K]:
				if j in ru:
					continue
				rank[j] += pi * wj
			return rank
		```
	- UserCF vs ItemCF
	![CF compare](https://raw.githubusercontent.com/hdu-dm-2017-movie-rs/recommending-system/master/CF%20compare.jpg)
#### 传统方法总结
  - 1.用户、物品、评分之间缺少实质性的关系，仅仅只是进行了相似度之间的计算
  - 2.矩阵运算时间长，矩阵稀疏，存在惰性学习，满足不了推荐系统的实时性和交互性
---

<br>

## 三、机器学习算法实现
  ![](E:/rs.jpg)
#### 1.提取训练集中用户观看的电影的ID，类型数据，评分，建立用户-电影类型-评分模型
#### 2.利用逻辑回归，随机森林等算法，训练出分类模型，得到用户的行为向量
#### 3.通过用户的历史数据和评分，预测用户可能会看的电影和电影类型
#### 4.将要推荐的电影与预测的电影数据做余弦相似度分析，得到要推荐的电影列表
---
## 四、接口文档
### movielens使用数据
  - ratings：userId，movieId，ratings，timestamp(未使用)
  - movies：movieId，movieName，genres
    - genres：'Action|Adventure|Animation|Children\'s|Comedy|Crime|Documentary|Drama|Fantasy|Film-Noir|Horror|IMAX|Musical|Mystery|Romance|Sci-Fi|Thriller|War|Western'
  - 数据预处理：删除 ratings 中的 `timestamp` 列


### reshape_train()：整理训练集，构建用户-电影矩阵
  - 使用 ratings + movies 构建训练集 data ，电影评分为5分制，3以下统一为0，3以上为1
  - 除去 `no genres listed` 电影类型
  - 加入用户历史数据
  - 构建矩阵，矩阵维数为[输入的用户历史条目数，19] (19为19个电影类型)
  - 输入：
    - 数据类型 array
    - 数据格式['movieName', 'movieId', 'rating', 'genres']
    - 需预先除去 `no genres listed` 电影类型
    - 电影评分 `rating` 为5分制
  - 输出：训练集矩阵 x_train, y_train

### user_matrix(array_input)：整理测试集，匹配矩阵，构建用户-电影矩阵
  - 测试集中 movieId 数据转成 `int.64` 类型，ratings 数据转成 `float64` 类型
  - 除去测试数据集 `movieId`，构建矩阵，矩阵维数为 [输入的用户历史条目数，19]
  - 输入：测试数据 数据来自`class Reshape`类的predict
    - 数据类型 array
    - 数据格式 ['movieName', 'movieId', 'rating', 'genres']
    - 需预先除去 `no genres listed` 电影类型
    - 电影评分 `rating` 为5分制
  - 输出：测试集矩阵 x_test, y_test

### calss MovieRS类：进行电影推荐
  - #### fit(x_train, y_train)：训练逻辑回归模型
  - 输入：训练集 x_train, y_train，训练集来自`class Reshape`类

  - #### predict(user, recommend, n=5)：预测测试集中用户可能看的电影
  - 输入：测试集 x_test，要推荐的电影数据 test_data，要推荐的电影数量 n
  - 输出：用户可能喜欢的电影，数据类型 array

  - #### CosineSim(x,y)：进行余弦相似度判定
  - 输入：1.要推荐的候选电影表
    - 数据类型 array
      - 数据格式 ['movieName', 'movieId', 'rating', 'genres']
      - 需预先除去 `no genres listed` 电影类型
      - 电影评分 `rating` 为5分制     
          2.用户可能喜欢的电影(来自predict的输出)
  - 输出：候选电影表和用户喜欢的电影特征之间进行比较，输出余弦相似度较高的电影
    - 数据类型 array
    - 数据格式 ['movieName', 'movieId', 'rating', 'genres']


### 调用示例
```py
  # 模型训练
  x_train, y_train = reshape_train()
  rs = MovieRS()
  rs.fit(x_train, y_train)
  
  # 推荐
  user_movie = rs.prediction(user, n=10)  //输入符合格式的用户数据
  result = rs.CosineSim(recommend, user_movie)  //输入符合格式的推荐电影表
  print(result)

```