

```python
import numpy as np
import pandas as pd
import os
import random
import sys
import math
from operator import itemgetter
ratings = os.path.join('ml-1m', 'ratings.dat')
data_ratings = open(ratings,'r')
```


```python
train = {}
test = {}
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

    train = 899999
    test = 100210
    


```python
#calculate movie to movie similarity
movie_pop = {}
movie_count = {}
for user, movies in train.items():
    for movie in movies:
        # count item popularity
        if movie not in movie_pop:
            movie_pop[movie] = 0
        movie_pop[movie] += 1
movie_count = len(movie_pop)
print movie_count
```

    3698
    


```python
# count co-rated users between items
movie_sim = dict()
for user, movies in train.items():
    for m1 in movies:
        for m2 in movies:
            if m1 == m2:
                continue
            movie_sim.setdefault(m1, {})
            movie_sim[m1].setdefault(m2, 0)
            movie_sim[m1][m2] += 1/math.log(1+len(movies)*1.0)
```


```python
# calculate similarity matrix v
simfactor_count = 0
PRINT_STEP = 2000000
for m1, related_movies in movie_sim.items():
    for m2, count in related_movies.items():
        movie_sim[m1][m2] = count / math.sqrt(movie_pop[m1] *movie_pop[m2])
        simfactor_count += 1
        if simfactor_count % PRINT_STEP == 0:
            print('calculating movie similarity factor(%d)' %simfactor_count)

```

    calculating movie similarity factor(2000000)
    calculating movie similarity factor(4000000)
    calculating movie similarity factor(6000000)
    calculating movie similarity factor(8000000)
    calculating movie similarity factor(10000000)
    


```python
#recommend
K = 20
N = 10
rank = {}
watched_movies = train[user]
for movie, rating in watched_movies.items():
    for related_movie, similarity_factor in sorted(movie_sim[movie].items(),key=itemgetter(1), reverse=True)[:K]:
        if related_movie in watched_movies:
            continue
        rank.setdefault(related_movie, 0)
        rank[related_movie] += similarity_factor * rating
# return the N best movies
recommend = sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]
print recommend
```

    [('260', 8.984853481495234), ('1240', 7.2425535288653675), ('1214', 6.529566986254497), ('589', 5.383357746651333), ('2916', 5.353802134412021), ('1200', 5.1542807430751445), ('1097', 4.767400149354729), ('2028', 4.484679303825415), ('480', 4.163039206652897), ('1580', 4.040902394538427)]
    


```python
#evaluate
hit = 0
rec_count = 0
test_count = 0
all_rec_movies = set()
# varables for popularity
popular_sum = 0

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

print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
       (precision, recall, coverage, popularity))

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
    precision=0.0393	recall=0.0237	coverage=0.0027	popularity=7.6537
    
