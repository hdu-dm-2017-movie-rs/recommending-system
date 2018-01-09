#### ml.py 程序说明
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