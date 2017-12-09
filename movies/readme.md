# 电影数据理解

所有格式都是CSV格式，如果不懂这种数据格式请自行搜索，可以把csv当作关系型数据库表。

以下是我个人对电影数据的文档简单整理。

目前的麻烦的地方在于，都是英文的电影数据，而我们电影推荐应该是各种中文信息，这点有点麻烦。

## ratings.csv

`用户`某个时间对`某电影`的`评分`，格式如下：
```
userId,movieId,rating,timestamp
1,31,2.5,1260759144
1,1029,3.0,1260759179
1,1061,3.0,1260759182
1,1129,2.0,1260759185
1,1172,4.0,1260759205
1,1263,2.0,1260759151
1,1287,2.0,1260759187
1,1293,2.0,1260759148
1,1339,3.5,1260759125
1,1343,2.0,1260759131
1,1371,2.5,1260759135
1,1405,1.0,1260759203
1,1953,4.0,1260759191
```

从0.5-5分的五分制的评分（可以考虑转换为1-10分的评分）

## tags.csv
`用户`在某个时间给`电影`打上`标签`，格式如下：
```
userId,movieId,tag,timestamp
15,339,sandra 'boring' bullock,1138537770
15,1955,dentist,1193435061
15,7478,Cambodia,1170560997
15,32892,Russian,1170626366
15,34162,forgettable,1141391765
15,35957,short,1141391873
15,37729,dull story,1141391806
15,45950,powerpoint,1169616291
15,100365,activist,1425876220
15,100365,documentary,1425876220
15,100365,uganda,1425876220
23,150,Ron Howard,1148672905
```

`标签`的种类很多，而且是英文的，目前不清楚怎么整合，豆瓣电影里面的标签应该是中文的吧？

## movies.csv

包含电影的`标题`和`类型`信息，格式如下：

```
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
4,Waiting to Exhale (1995),Comedy|Drama|Romance
5,Father of the Bride Part II (1995),Comedy
6,Heat (1995),Action|Crime|Thriller
7,Sabrina (1995),Comedy|Romance
8,Tom and Huck (1995),Adventure|Children
9,Sudden Death (1995),Action
10,GoldenEye (1995),Action|Adventure|Thriller
11,"American President, The (1995)",Comedy|Drama|Romance
12,Dracula: Dead and Loving It (1995),Comedy|Horror
13,Balto (1995),Adventure|Animation|Children
14,Nixon (1995),Drama
```

`Genres` 是用竖线分割的列表，均是以下值:

- Action
- Adventure
- Animation
- Children's
- Comedy
- Crime
- Documentary
- Drama
- Fantasy
- Film-Noir
- Horror
- Musical
- Mystery
- Romance
- Sci-Fi
- Thriller
- War
- Western
- (no genres listed)


## links.csv

该文件用于链接到其他电影数据源，格式如下：

```
movieId,imdbId,tmdbId
1,0114709,862
2,0113497,8844
3,0113228,15602
4,0114885,31357
5,0113041,11862
6,0113277,949
7,0114319,11860
8,0112302,45325
9,0114576,9091
10,0113189,710
11,0112346,9087
12,0112896,12110
13,0112453,21032
14,0113987,10858
```

movieId is an identifier for movies used by https://movielens.org. E.g., the movie Toy Story has the link https://movielens.org/movies/1.

imdbId is an identifier for movies used by http://www.imdb.com. E.g., the movie Toy Story has the link http://www.imdb.com/title/tt0114709/.

tmdbId is an identifier for movies used by https://www.themoviedb.org. E.g., the movie Toy Story has the link https://www.themoviedb.org/movie/862.

Use of the resources listed above is subject to the terms of each provider.

## 其他

### Cross-Validation

Prior versions of the MovieLens dataset included either pre-computed cross-folds or scripts to perform this computation. We no longer bundle either of these features with the dataset, since most modern toolkits provide this as a built-in feature. If you wish to learn about standard approaches to cross-fold computation in the context of recommender systems evaluation, see LensKit for tools, documentation, and open-source code examples.