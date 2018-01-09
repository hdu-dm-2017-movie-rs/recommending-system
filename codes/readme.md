### movielens 推荐算法说明
- 使用ml-latest数据做训练和测试，使用glove.6B.100d做词向量预处理模型
- 使用numpy，pandas，sklearn，matplotlib，keras扩展
- 相关文档：ml-Word Embeddings and Deep Learning.ipynb
  ml-lstm.ipynb
#### 数据预处理
- 数据格式
  - ratings：userid，movieid，rating，timestamp(不使用)
  - tags：userid，movieid，tag，timestamp
  - movies：movieid，title，genres
```py
  # Load datasets
  ratings = pd.read_csv('ml-latest/ratings.csv')
  print ('Shape of the ratings data frame:', ratings.shape)
  ratings = ratings.drop(['timestamp'],axis=1)
  tags = pd.read_csv('ml-latest/tags.csv')
  print ('Shape of the tags data frame:', tags.shape)

  movies = pd.read_csv('ml-latest/movies.csv')
  print ('Shape of the movies data frame:', movies.shape)

  #Will take
  tags = tags.sample(frac=0.2)
  ratings = ratings.sample(frac=0.2)
```
- 使用`ratings`+`tag`的数据：`data = pd.merge(ratings, tags, how='inner')`

#### 数据清理，优化
- 查看电影的评分分布并进行0/1规划
- 清除特殊标签
- 将时间改为datetime格式
```py
  data['rating'] = data['rating'].apply(lambda x: 1 if x > 4 else 0)

  data['tag'] = data['tag'].apply(lambda x: str(x))
  data['tag'] = data['tag'].map(lambda x: re.sub(r'([^\s\w]|_)+', '', x))
  data['tag'] = data['tag'].str.lower()

  data['timestamp'] = data['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
  data['timestamp'].astype('datetime64[ns]')[0:5]
```
#### 降维处理
- 利用词向量处理`tags`数据集中的评论，先预载入Stanford's global word embeddings 作为预先训练模型
```py
  embeddings_index = {}
  f = open('glove.6B/glove.6B.100d.txt')
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()

  print('Found %s word vectors.' % len(embeddings_index))

  embedding_matrix = np.zeros((len(words), 100))
  for i in range(len(words)):
      embedding_vector = embeddings_index.get(words[i])
      if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
          embedding_matrix[i] = embedding_vector
          
  pdembedding = pd.DataFrame(embedding_matrix.T,columns=words)
```
- 进行PCA降维，k-means聚类并输出可视化结果
- 训练完成后载入之前处理好的`data`
```py
  #Cluster word embeddings data 
  kmeans = KMeans(init='k-means++', n_clusters=300, n_init=10)
  kmeans.fit(pdembedding.T)
  #Get cluster labels
  clusters = kmeans.labels_

  #Add columns to data for each cluster
  for i in range(max(clusters)+1):
  data[i] = 0

  #If word is in data row, label the associated cluster accordingly with 1
  for i in range(len(pdwordvec.columns)):
      column = pdwordvec.columns[i]
      index = pdwordvec[column].loc[pdwordvec[column] > 0, ].index
      for ii in range(len(index)):
          data.loc[index[ii],clusters[i]] = 1
```
- 增加了`movies`数据中的`genre`元素，对movieid，userid，genre进行`one-hot`编码处理
```py
  #Drop tag as we will use vectorized words
  data = data.drop(['tag'], axis=1)

  #Add genres
  #Split genre column
  genresplit = movies.set_index('movieId').genres.str.split(r'|', expand=True).stack().reset_index(level=1, drop=True).to_frame('genre')
  #Use one-hot encoding grouped by genre 
  genres = pd.get_dummies(genresplit, prefix='genre', columns=['genre']).groupby(level=0).sum()
  #Drop unnecessary field, if all genres are 0 then it means no genres are listed. 
  genres = genres.drop(['genre_(no genres listed)'], axis=1)
  #Join data by movieId
  genres['movieId'] = genres.index
  data = pd.merge(data, genres, on='movieId', how='left')

  #Assign variables as categorical using one hot encoding
  useridencoding = pd.get_dummies(data['userId'], prefix='userid')
  data = data.drop(['userId'], axis=1)
  data = pd.concat([data, useridencoding], axis=1)

  movieidencoding = pd.get_dummies(data['movieId'], prefix='movieid')
  data = data.drop(['movieId'], axis=1)
  data = pd.concat([data, movieidencoding], axis=1)
  data = data.fillna(np.nan)
```

#### 使用传统的模型进行训练
- 设置训练集和数据集
- 修改了训练集和数据集的分配策略，可显著提升分类效果
```py
  #train = data[(data['timestamp'] < '2016-08-01') ]
  #test = data[(data['timestamp'] >= '2016-08-01') ]
  from sklearn.model_selection import train_test_split
  train, test = train_test_split(data, test_size=0.2, random_state=0)
  #随机选择20%作为测试集，剩余作为训练集
  train = train.drop(['timestamp'], axis=1)
  test = test.drop(['timestamp'], axis=1)
  
  y_train = train['rating']
  y_test = test['rating']
  x_train = train.drop(['rating'], axis=1)
  x_test = test.drop(['rating'], axis=1)
```
- 基准：0.5754->0.6135
- Logistic Regression：0.6534->0.7583
```py
  logreg = LogReg(C = 1, class_weight='balanced')
  logreg.fit(x_train, y_train)
  y_predlog = logreg.predict_proba(x_test)
  R2_log = logreg.score(x_test,y_test) 
  print "Accuracy of the test set for log. reg. is: ", np.round(R2_log,4)
```
- Random Forest：0.6547->0.7693
```py
  RFC = RandomForestClassifier(class_weight='balanced')
  RFC.set_params(n_estimators=100)
  RFC.fit(x_train,y_train)
  R2_rfc = RFC.score(x_test,y_test) 
  print "Accuracy of the test set for random forest is: ", np.round(R2_rfc,4)
```
- MultinomialNB：0.6610->0.758
```py
  NB = MultinomialNB()
  NB.fit(x_train,y_train)
  R2_nb = NB.score(x_test,y_test) 
  print "Accuracy of the test set for Multinomial NB model is: ", np.round(R2_nb,4)
```
- AdaboostClassifier：0.6079->0.6462
```py
  AdaB = AdaBoostClassifier()
  AdaB.fit(x_train,y_train)
  R2_ada = AdaB.score(x_test,y_test) 
  print "Accuracy of the test set for AdaBoost model is: ", np.round(R2_ada,4)
```
- xgboost：0.64->0.689
```py
  clf = xgb.XGBClassifier(max_depth=20, learning_rate=0.1) 
  clf.fit(x_train, y_train, early_stopping_rounds=10, eval_set=[(x_train, y_train), (x_test, y_test)])  
  R2_xgb = clf.score(x_test,y_test) 
  print "Accuracy of the test set for XGBoost model is: ", np.round(R2_xgb,2)
```

#### 使用深度学习进行训练
- 构建训练集和测试集
```py
  dpdata = pd.concat([data, pdseq], axis=1)
  dpdata = dpdata.drop(['tag'], axis=1)
  dpdata = dpdata.drop(['userId'], axis=1)
  dpdata = dpdata.drop(['movieId'], axis=1)

  #train = dpdata[(dpdata['timestamp'] < '2016-08-01') ]
  #test = dpdata[(dpdata['timestamp'] >= '2016-08-01') ]
  train, test = train_test_split(data, test_size=0.2,random_state=0)
  #随机选择20%作为测试集，剩余作为训练集
  train = train.drop(['timestamp'], axis=1)
  test = test.drop(['timestamp'], axis=1)
  y_train = train['rating']
  y_test = test['rating']
  x_train = train.drop(['rating'], axis=1)
  x_test = test.drop(['rating'], axis=1)
```
- GRU,LSTM
```py
  y_test_matrix = to_categorical(y_test)
  y_train_matrix = to_categorical(y_train)
  x_train_array = np.array(x_train)
  x_test_array = np.array(x_test)

  epochs = 20
  lrate = 0.001
  sgd = SGD(lr=lrate)
  early_stopping = EarlyStopping(monitor='acc',patience=2)

  model = Sequential()
  model.add(Embedding(len(word_index)+1, 100, mask_zero=True, trainable=False))
  model.add(GRU(10, return_sequences=False))
  #model.add(LSTM(10, return_sequences=False))
  model.add(Dense(2, activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
  model.summary()
```

#### usercf.ipynb/item.ipynb
- `usercf`和`itemcf`代码，在jupyter notebook上初步实现
- 目前使用了 movie-lens 1 million 数据级中的`ratings.dat`做分析
- 计算 co-relation 部分非常耗时，需要重点做一下优化
- `itemcf` 中 `recommend` 部分出现参数 `rank > 1.0` 的情况，需要继续解决
- 准确率，召回率，覆盖率都不理想，需要进一步做综合推荐进行优化