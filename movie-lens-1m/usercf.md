

```python
import numpy as np
import pandas as pd
import os
import random
import sys
import math
from operator import itemgetter

#ml-1m数据集读取 
#data = os.path.join( os.path.expanduser("E:\WPX\study\AI research\DM\data"),"ml-1m")  
#ratings = os.path.join( data,"ratings.dat")
#data_ratings = pd.read_csv(ratings, delimiter="::",header=None, names=["UserID","MovieID","Rating","Datetime"])   
#时间格式转换  
#data_ratings["Datetime"]=pd.to_datetime(data_ratings["Datetime"],unit='s')  
ratings = os.path.join('ml-1m', 'ratings.dat')
data_ratings = open(ratings,'r')
```


```python
#load rating data and split it to training set and test set
train = {}#训练集
test = {}#测试集
train_len = 0
test_len = 0
pivot=0.9
for line in data_ratings:
    user, movie, rating, _ = line.split('::')
    # split the data by pivot
    if random.random() < pivot:
        train.setdefault(user, {})
        train[user][movie] = int(rating)
        train_len += 1
    else:
        test.setdefault(user, {})
        test[user][movie] = int(rating)
        test_len += 1
print ('train = %s' % train_len)
print ('test = %s' % test_len)
```

    train = 899718
    test = 100491
    


```python
#calculate movie_users similarity
movie_user = {}
movie_pop = {}
movie_count = {}
for user, movies in train.items():
    for movie in movies.keys():
        # inverse table for movie_users
        if movie not in movie_user:
            movie_user[movie] = set()
        movie_user[movie].add(user)
        # count item popularity at the same time
        if movie not in movie_pop:
            movie_pop[movie] = 0
        movie_pop[movie] += 1
movie_count = len(movie_user)
print movie_count
```

    3692
    


```python
# count co-rated movies between users
#计算特别耗时间
user_sim = {}#构建user间movies的相关性字典
for movie, users in movie_user.items():
    for u in users:
        for v in users:
            if u == v:
                continue
            user_sim.setdefault(u, {})
            user_sim[u].setdefault(v, 0)
            user_sim[u][v] += 1/math.log(1+len(users))
```


```python
# calculate similarity matrix w
sim_count = 0
PRINT_STEP = 2000000
for u, related_users in user_sim.items():
    for v, count in related_users.items():
        user_sim[u][v] = count/math.sqrt(len(train[u]) * len(train[v]))
        sim_count += 1
        #用于查看进度
        if sim_count % PRINT_STEP == 0:
            print ('calculating user similarity factor(%d)' % sim_count)
```

    calculating user similarity factor(2000000)
    calculating user similarity factor(4000000)
    calculating user similarity factor(6000000)
    calculating user similarity factor(8000000)
    calculating user similarity factor(10000000)
    calculating user similarity factor(12000000)
    calculating user similarity factor(14000000)
    calculating user similarity factor(16000000)
    calculating user similarity factor(18000000)
    calculating user similarity factor(20000000)
    calculating user similarity factor(22000000)
    calculating user similarity factor(24000000)
    calculating user similarity factor(26000000)
    calculating user similarity factor(28000000)
    calculating user similarity factor(30000000)
    calculating user similarity factor(32000000)
    calculating user similarity factor(34000000)
    


```python
#recommend
K = 20#取相似度排前20的
N = 10#前10的推荐
rank = dict()
watched_movies = train[user]
for similar_user, similarity_factor in sorted(user_sim[user].items(),key=itemgetter(1), reverse=True)[0:K]:
    for movie in train[similar_user]:
        if movie in watched_movies:
            continue
# predict the user's interest for each movie
        rank.setdefault(movie, 0)
        rank[movie] += similarity_factor
# return the N best movies
recommend = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]
print recommend
```

    [('260', 0.7787905460429678), ('1214', 0.6884310151266817), ('589', 0.6468363960267578), ('32', 0.6023442678502415), ('480', 0.5938990210217743), ('1206', 0.5938706308558006), ('1200', 0.5920882180419209), ('1240', 0.5878607166359372), ('1580', 0.555366186588709), ('1653', 0.5513193015747229)]
    


```python
#evaluate
#准确率，召回率，覆盖率，流行度
hit = 0
rec_count = 0
test_count = 0
popular_sum = 0
all_rec_movies = set()

for i, user in enumerate(train):
    if i % 500 == 0:
        print ('recommended for %d users' % i)
    test_movies = test.get(user, {})
    rec_movies = recommend
    for movie, _ in rec_movies:
        if movie in test_movies:
            hit += 1
        all_rec_movies.add(movie)
        popular_sum += math.log(1 + movie_pop[movie])
    rec_count += N
    test_count += len(test_movies)

precision = hit / (1.0 * rec_count)
recall = hit / (1.0 * test_count)
coverage = len(all_rec_movies) / (1.0 * movie_count)
popularity = popular_sum / (1.0 * rec_count)

print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' % (precision, recall, coverage, popularity))
```

    recommended for 0 users
    recommended for 500 users
    recommended for 1000 users
    recommended for 1500 users
    recommended for 2000 users
    recommended for 2500 users
    recommended for 3000 users
    recommended for 3500 users
    recommended for 4000 users
    recommended for 4500 users
    recommended for 5000 users
    recommended for 5500 users
    recommended for 6000 users
    precision=0.0333	recall=0.0200	coverage=0.0027	popularity=7.4769
    
