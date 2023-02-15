# -*- coding: utf-8 -*-

# -- HW4 --

# <div class="alert alert-block alert-success">
#     <b><h2>Условие 1: </h2>
#     <h3>Задача 1</h3>
#     <br>- Постройте график
#     <br>- Назовите график
#     <br>- Сделайте именование оси x и оси y
#     <br>- Сделайте выводы
#     <br>
#     <br>1.1. Скачать данные по ссылке https://www.kaggle.com/datasets/esratmaria/house-price-dataset-with-other-information </b>
# </div>


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline

import plotly.express as px
import tensorflow
import gmaps
import gmaps.datasets

df = pd.read_csv(r'kc_house_data.csv', encoding='cp1251')
df

max_lat=47.1559; min_lat=47.7776
max_long=-121.315; min_long=-122.519

# <div class="alert alert-block alert-success">
#     <b>1.2 Изучите стоимость недвижимости</b>
# </div>  


plt.hist(df['price'])
plt.title("Стоимость недвижимости")
plt.xlabel("Цена")
plt.ylabel("Количество")

plt.figure(figsize=(6, 4))

plt.scatter(df['price'], df['zipcode'])

plt.title("Зависимость стоимости недвижимости от места нахождения")
plt.xlabel("Цена")
plt.ylabel("Индекс")

pr = df['price'].value_counts()
pr

# <div class="alert alert-block alert-success">
#     <b>1.3 Изучите распределение жилой квадратуры</b>
# </div>  


plt.hist(df['sqft_living'])
plt.title("Распределение жилой квадратуры")
plt.xlabel("Жилая квадратура")
plt.ylabel("Количество")

sql = df['sqft_living'].value_counts()
sql

# <div class="alert alert-block alert-success">
#     <b>1.4 Изучите распределение года постройки</b>
# </div>  


plt.hist(df['yr_built'])
plt.title("Распределение года постройки")
plt.xlabel("Года постройки")
plt.ylabel("Количество")

yrb = df['yr_built'].value_counts()
yrb

plt.bar(yrb.index, yrb.values)
plt.title("Распределение года постройки")
plt.xlabel("Года постройки")
plt.ylabel("Количество")

# <div class="alert alert-block alert-success">
#     <b><h2>Условие 2: </h2>
#     <h3>Задача 2</h3>
#     <br>2.1 Изучите распределение домов от наличия вида на набережную
#     <br><br>- Постройте график
#     <br>- Сделайте выводы</b>
# </div> 


wf = df['waterfront'].value_counts()
plt.figure(figsize=(6, 4))

plt.pie(wf.values, autopct='%.1f%%')

plt.title("Распределение домов от наличия вида на набережную")
plt.legend(wf.index)

# <div class="alert alert-block alert-success">
#     <b>2.2 Изучите распределение этажей домов</b>
# </div>  


plt.hist(df['floors'])
plt.title("Распределение этажей домов")
plt.xlabel("Этажность")
plt.ylabel("Количество")

fl = df['floors'].value_counts()
fl

plt.bar(fl.index, fl.values)
plt.title("Распределение этажей домов")
plt.xlabel("Этажность")
plt.ylabel("Количество")

fl = df['floors'].value_counts()
plt.figure(figsize=(6, 4))

plt.pie(fl.values, autopct='%.1f%%')

plt.title("Распределение этажей")
plt.legend(fl.index)

# <div class="alert alert-block alert-success">
#     <b>2.3 Изучите распределение состояния домов</b>
# </div>  


cond = df['condition'].value_counts()
cond

plt.bar(cond.index, cond.values)
plt.title("Распределение состояния домов")
plt.xlabel("ценка состояния")
plt.ylabel("Количество")

plt.pie(cond.values, autopct='%.1f%%')
plt.legend(cond.index)
plt.title("Распределение состояния домов")

# <div class="alert alert-block alert-success">
#     <b><h2>Условие 3: </h2>
#     <h3>Задача 3</h3>
#    <br>Исследуйте, какие характеристики недвижимости влияют на стоимость недвижимости, с применением не менее 5 диаграмм из урока.
#     <br>Анализ сделайте в формате storytelling: дополнить каждый график письменными выводами и наблюдениями.</b>
# </div>  


# Исследуем распределение целевых переменных относительно цены :


f, ax = plt.subplots(2, 2, figsize=(16, 16))
plt.subplots_adjust(left=None, bottom=1.5, right=None, top=3, wspace=None, hspace=None)
col = 3
i = 1
while i < 4:
    j = 1
    while j < 7:
        plt.subplot(6, 3, col - 2)
        plt.scatter(x=df.loc[:, df.columns[col]], y=df.price)
        plt.xlabel(df.columns[col])
        plt.ylabel("Price")
        col += 1
        j += 1
    i += 1

# На графиках выше отображаются зависимости различных параметров от цены:
# напимер, люди больше предпочитают селиться на средних этажах, а цена напрямую зависит от жилой площади и оценки.


# Зависимость грейда относительно цены и жилого метража:


wf = df['waterfront']
fig = px.scatter(df, x='sqft_living', y='price', color='grade', symbol='condition', size='waterfront')
fig

# Зависимость оценки состояния относительно жилого метража и цены


fig = px.scatter(df, x='sqft_living', y='yr_built', color='condition', size='price',
                 color_continuous_scale=["yellow",
                                         "green", "blue",
                                         "purple"])

fig

multi_df = df[1:4]
multi_df

# Расположение домов на карте позволяет не делать выводы по зависимости стоимости от расположения недвижимости:


import folium
house_map = folium.Map(prefer_canvas=True)

def plotDot(point):
    folium.CircleMarker(
        location=[point.lat, point.long],
        radius=2,
        popup=point.price
    ).add_to(house_map)


df.apply(plotDot, axis=1)

house_map.fit_bounds(house_map.get_bounds())

house_map.save('house_map.html')

# Так как карта получается слишком тяжелой, я выгрузила ее в html, который прилагаю, в так же ниже вставляю часть данной карты в виде картинки ниже:


from PIL import Image
img = Image.open(r'house_map.jpg')
img.show()

# На данном графике можно отметить три города:


sns.jointplot(x=df['lat'], y=df['long'], kind='kde');

