---
layout: post
title: Plot Geospatial Points in Python
author: jongsukim
date: 2021-08-14 12:00:00 +0900
categories: [Programming, Python]
tags:
  - GeoPandas
  - Matplotlib
  - Geospatial Data
  - Data Visualization
math: false
---

학위 논문 심사 발표 준비를 하면서 introduction용으로 지도에 point를 찍어서 표현할 필요가 있었다. 그것도 한중일 지도와 서울 지도를 같이 보여줘야 했기에 두 지도를 동시에 보여줄 필요가 있었다. 즉 다음 조건을 만족하는 그림을 그리고자 했다.

* 두 지도를 한번에 보여줄 수 있어야 한다.
* point는 위도(latitude)와 경도(longitude)로 주어진다.

Python으로 그린 그 과정을 정리한 글이다. 아래의 모든 코드는 [Google Colab](https://colab.research.google.com/drive/1piFRokmhxutjR1H4ZzkldFCrveYEo-xg?usp=sharing)에서 확인할 수 있다.
변수 네이밍이 좀 많이 구린데 (0, 1 인덱스의 오용 등등) 당시 급하게 짰던거라 양해바란다.

## Data

당연히 데이터가 필요하다. 그리고 패키지가 필요하다. 여러개 찾아봤는데 러닝커브 짧고 (빨리 만들어야해서 금방 가져다 쓸수 있는게 필요했다), 문서화 잘 되어있던 걸 찾다가 [GeoPandas](https://geopandas.org/index.html)를 고르게 되었다. 포맷 적당하고, matplotlib랑 호환도 잘 돼서 내가 쓰기 편했다. 또 NUMFOCUS에서 지원받으니 어느정도 maintain되는 패키지이지 않을까 생각했다.

이제 포맷을 정해야하는데 공간 정보 데이터를 다루는 포맷이 여러개가 있다. GeoPandas에서는 Shapefile, GeoJSON, GeoPackage를 지원하는 것 같다. 나는 JSON이 편하니깐 GeoJSON을 골랐다. 예전에 Shapefile써봤는데 너무 어려웠다. 간단하게 윤곽만 보이면 돼서 그냥 편한거 쓰자 하는 생각에 GeoJSON을 골랐다. TopoJSON이 더 컴팩트하고 좋은거 같은데, 데이터 구하기가 힘들었다. 어차피 파일 다운받아서 그릴건데 용량이 뭔 상관인가 싶어서 그냥 GeoJSON을 쓰기로 했다.

한중일(CJK) 데이터는 [DataHub](https://datahub.io/core/geo-countries)라는 곳에서, 서울시 데이터는 [seoul-maps](https://github.com/southkorea/seoul-maps/)에서 구했다. 한중일만 따로 있는게 아니고, 전세계의 지도 데이터이기 때문에 실질적으로 사용할 때는 위도와 경도의 범위 제한을 통해서 한중일만 plot하면 된다.

## Load Data

심플하다. 그냥 [`read_file`](https://geopandas.org/docs/reference/api/geopandas.read_file.html)에 url이든 파일 이름이든 넣으면 알아서 파싱해서 가져온다.

```
import geopandas as gpd

# download countries geojson data
cjk_url = 'https://datahub.io/core/geo-countries/r/countries.geojson'
cjk_df = gpd.read_file(cjk_url)

# download seoul geojson data
seoul_url = 'https://github.com/southkorea/seoul-maps/raw/master/kostat/2013/json/seoul_municipalities_geo_simple.json'
seoul_df = gpd.read_file(seoul_url)
```

## Plot CJK Map
GeoPandas 자체적으로 [plot](https://geopandas.org/docs/reference/api/geopandas.GeoDataFrame.plot.html)함수를 지원하기 때문에 plot해주면 된다.
API 문서를 보면 나오듯이, matplotlib axes instance로 return이 되기 때문에 한번 그리고 나면 나머지는 matplotlib만 생각하면 된다.

```Python
# plot CJK
ax = cjk_df.plot(figsize=(7.22, 6.22), alpha=0.8, color='#fff',
                    edgecolor='#777')
ax.set_facecolor('#add8e6')
fig = plt.gcf()
ax.set_xlim((116, 132))
ax.set_ylim((32, 45))
ax.set_aspect(1.0)
```

바다를 그리기 위해서 facecolor를 푸른색 계열로 지정해주었다. 이부분이 좀 어려운데, 덧칠의 개념이라고 생각하면 편하다.

1. 흰색(`color='#fff'`)에 alpha값을 0.8(80%의 투명도)로 설정한다.
2. 위를 `#add8e6`으로 덧칠한다. (`ax.set_facecolor('#add8e6')`)

저렇게 육지와 바다가 구분된 색으로 이쁘게 나온다. 색은 아마 구글 맵 색을 따왔던 걸로 기억한다. 여기서 중요한건 `plot` 함수 옵션에 `facecolor`를 넣으면 안되고, `ax.set_facecolor`로 따로 코드를 작성해야 잘 나온다. 이유는 사실 잘 모르겠다.

그리고 x축과 y축 범위를 경도와 위도를 참고해서 적절히 설정하면 우리나라가 가운데 있으면서 중국과 일본이 일부분 보이는 그런 그림이 나온다.

![China-Korea-Japan Image](/assets/images/post/2021-08-14-Geospatial/CJK.png)

## Plot Seoul Map

요동 반도 근처에 박스를 그려서 그 안에 서울 지도를 넣으려고 한다. 다음과 같은 프로세스를 밟는다.

1. 서울의 실제 **위치**를 그리기 위해 `matplotlib`의 [`Rectangle`](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html)로 작은 박스를 생성
2. 서울의 **확대된 맵**을 그리기 위해 CJK 맵을 그렸던 Axes에 [`inset axes`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.inset_axes.html)로 큰 박스를 그리고 서울 지도를 임베딩

그러면 다음과 같은 코드를 사용하면 된다.

```Python
seoul_lat = [126.83, 127.09]
seoul_lon = [37.5, 37,57]

# create small box(Rectangle) for Seoul
seoul_sbox = (seoul_lat[0], seoul_lon[0])
seoul_lbox = (117, 38)
lbox_size = 6
sbox_size = 0.3

rect = Rectangle((seoul_lat[0], seoul_lon[0]),
                    sbox_size, sbox_size,
                    linewidth=0.5, edgecolor='k', facecolor='white', zorder=6)
ax.add_artist(rect)

# create large box(inset) for zoomed Seoul
axin_seoul = ax.inset_axes(bounds=[seoul_lbox[0], seoul_lbox[1],
                                   lbox_size, lbox_size],
                           transform=ax.transData, alpha=0.4, zorder=6)

# plot Seoul
seoul_df.plot(ax=axin_seoul, color='none',
                edgecolor='#333', facecolor='none', alpha=0.3, zorder=6)
```

### Small Box

서울의 실제 위치를 그리는 작은 박스를 그릴 것이다.
`Rectangle`의 첫번째 전달인자로는 `anchor point`를 지정한다.
`Rectangle`의 `anchor point`는 "일반적으로" 박스의 왼쪽 아래의 좌표를 뜻한다.
서울의 적절한 위도와 경도 범위를 `seoul_lat`, `seoul_lon`이라 정의하고,
경도와 위도의 크기로 0.3도 정도의 박스를 그린다고 가정하고 이를
`sbox_size`라는 변수로 지정하였다. 즉 경도상으로는 126.83°부터 127.13°까지, 위도상으로는 37.5°부터 37.8°까지를 그린다.

### Large Box

이제 실제 서울 지도를 지도에 표시할 차례이다. 요동반도 근처 적당한 크기 (6도)의 박스를 그릴 예정이고,
위치를 `seoul_lbox` 변수, 그리고 크기를 `lbox_size`라는 변수에 대입하였다.

`Rectangle`과는 다르게 기존 `Axes`에 또다른 `plot`이 추가되는 개념이기 때문에 matplotlib의 [`inset axes`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.inset_axes.html)을 사용한다. `Rectangle`과는 다르게 `anchor point`와 `box size`를 `bounds=[117, 38, 6, 6]`에 한번에 넣는데, 경도상으로는 117°부터 121°까지, 위도상으로는 38°부터 44°까지를 그린다.을 지정한다. 이때 위도와 경도를 넣기 위해 좌표계산을 좌표의 상대적인 비율이 아니라 데이터의 절대적인 값으로 인식할 수 있게 해야한다. 그러기에 `transform=ax.transData` 또한 전달인자로 넣는다. `transform`에 대한 것은 [공식 문서](https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html)에 설명이 잘 되어 있다.

이렇게 생성한 `inset`을 `axin_seoul`라는 변수로 지정하고, 이를 `geopandas`의 plot함수에서 `ax` 전달인자로 넣는다. 그 결과는 다음과 같다. `inset`의 위치를 지정하는 방법은 [데모 페이지](https://matplotlib.org/stable/gallery/axes_grid1/inset_locator_demo.html)가 설명이 잘 되어있다.

![China-Korea-Japan and Seoul Image](/assets/images/post/2021-08-14-Geospatial/CJK-Seoul.png)

### Draw Lines between Small Box and Large Box

위의 그림으로 끝내면 작은 박스(`Rectangle`)와 큰 박스(`inset`)의 관계를 알기가 어렵다. 따라서 둘을 직선으로 이어서 작은 박스를 확대한 것이 큰 박스임을 나타내고자 한다.
이는 다음과 같은 코드로 그릴 수 있다.

```Python
# connect rect to inset
x0, y0 = (seoul_lbox[0], seoul_lbox[1] + 0.5)
x1, y1 = (seoul_lbox[0] + lbox_size, seoul_lbox[1] + lbox_size - 0.5)
px0, py0 = (seoul_sbox[0], seoul_sbox[1])
px1, py1 = (seoul_sbox[0] + sbox_size, seoul_sbox[1] + sbox_size)
verts_0 = [(px0, py0), (x0, y0), (px0, py0)]
verts_1 = [(px1, py1), (x1, y1), (px1, py1)]
codes_0 = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
codes_1 = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]

path_0 = mpath.Path(verts_0, codes_0)
path_1 = mpath.Path(verts_1, codes_1)

patch_0 = ax.add_patch(mpatches.PathPatch(path_0, facecolor='k', lw=0.5))
patch_1 = ax.add_patch(mpatches.PathPatch(path_1, facecolor='k', lw=0.5))
```

우선 선은 `[matplotlib.path.Path](https://matplotlib.org/stable/api/path_api.html)`를 사용하여 그린다. 그리고 [튜토리얼](https://matplotlib.org/stable/tutorials/advanced/path_tutorial.html)이 매우 많은 도움이 되었다.

선은 두 점을 잇는다고 할 수 있지만, `matplotlib`에서의 `Path`는 두 점을 **왕복** 한다고 생각하였으며, 마지막으로 `CLOSEPOLY`를 code로 설정한다고 생각하였다.
시작점으로 이동하기 위해 `MOVETO`, 목표하는 점으로 선을 긋기 위해 `LINETO`, 다시 돌아와서 마무리 짓기 위해 `CLOSEPOLY`, 이렇게 세 가지의 코드를 지정한다. 이것이 `matplotlib.path.Path`의 두번째 전달인자에 들어가는 코드이다.

`matplotlib.path.Path`의 첫번째 인자인 `vertex`는 각 포인트를 뜻한다. 이는 큰 박스에서 작은 박스로 설정했는데, 이건 순서는 상관없는 것 같다. 코드가 복잡해보이는데, 큰 박스의 왼쪽 아래(`x0, y0`)에서 작은 박스의 왼쪽 아래(`px0, py0`), 그리고 큰 박스의 오른쪽 위(`x1, y1`)에서 작은 박스의 오른쪽 위(`px1, py1`)로 설정한 것 뿐이다.

이렇게 생성한 `mpatches.PathPatch`를 통해 `Path`를 `Patch`로 변환하고, 이를 `ax.add_patch`를 불러와서 원래의 맵에 추가하면 된다.

![China-Korea-Japan and Seoul Image with line](/assets/images/post/2021-08-14-Geospatial/CJK-Seoul-line.png)

# Draw Points in Seoul Map

Dictionary로 된 points를 서울 지도에 표시해 줄 필요가 있다.

Aspect ratio를 고려하여 원으로 그리기 위해 [`matplotlib.patches.Ellipse`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.Ellipse.html)를 사용하였다.
실제 점은 원처럼 보여야하지만 지도 자체의 aspect ratio가 1이 아니었기 때문에 타원으로 그리고 비율을 조정한 것이다. 전달인자는 앞에서의 도형들과 같이 위치와 크기, 그리고 색 등의 전달인자이다. 종로구, 강서구와 서초구에는 두 배 크고, 색도 다른 색으로 지정하였다.

```Python
stations_latlon = {
        "중구" : [126.9747, 37.5643],
        "종로구" : [127.0050, 37.5720],
        "용산구" : [127.0048, 37.5400],
        "광진구" : [127.0925, 37.5472],
        "성동구" : [127.0419, 37.5432],
        "중랑구" : [127.0940, 37.5849],
        "동대문구" : [127.0289, 37.5758],
        "성북구" : [127.0273, 37.6067],
        "도봉구" : [127.0290, 37.6542],
        "은평구" : [126.9348, 37.6098],
        "서대문구" : [126.9378, 37.5767],
        "마포구" : [126.9456, 37.5498],
        "강서구" : [126.8351, 37.5447],
        "구로구" : [126.8897, 37.4985],
        "영등포구" : [126.8974, 37.5250],
        "동작구" : [126.9715, 37.4809],
        "관악구" : [126.9271, 37.4874],
        "강남구" : [127.0481, 37.5176],
        "서초구" : [126.9945, 37.5046],
        "송파구" : [127.1165, 37.5218],
        "강동구" : [127.1368, 37.5450],
        "금천구" : [126.9083, 37.4524],
        "강북구" : [127.0288, 37.6379],
        "양천구" : [126.8587, 37.5234],
        "노원구" : [127.0679, 37.6574]}

point_r = 0.012
aspect = axin_seoul.get_aspect()

for station, loc in stations_latlon.items():
    lat, lon = loc[0], loc[1]
    p = Ellipse((lat, lon), point_r, point_r / aspect, fc='#1A4E66', zorder=7)
    if station == '강서구' or station == '서초구':
        p = Ellipse((lat, lon), 2.0 * point_r, 2.0 * (point_r / aspect),
                    fc='#E26C22', zorder=7)
    if station == '종로구':
        p = Ellipse((lat, lon), 2.0 * point_r, 2.0 * (point_r / aspect),
                    fc='#00A1F1', zorder=7)
    axin_seoul.add_artist(p)
```

![China-Korea-Japan and Seoul Image with points](/assets/images/post/2021-08-14-Geospatial/CJK-Seoul-points.png)

## Hide Axis

위도, 경도가 꼭 표시되어야 할 필요가 없는 정보였기 때문에 axis자체를 숨기기로 하였다.

```Python
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
axin_seoul.xaxis.set_visible(False)
axin_seoul.yaxis.set_visible(False)
plt.tight_layout()
```

짜잔! 다음과 같이 깔끔하게 그려진 서울이 확대된 한중일 지도가 그려졌다!

![China-Korea-Japan and Seoul Final Image](/assets/images/post/2021-08-14-Geospatial/CJK-Seoul-final.png)

## Conclusion

위 방법의 핵심은 다음과 같다.

1. Geospatial한 데이터를 어떻게 그릴것인가? -> `GeoJSON`을 `GeoPandas`를 통해서 사용
2. 어떻게 두 지도를 한 Figure에 그릴 수 있는가? -> `matplotlib`의 `inset` 사용
3. 어떻게 선과 포인트를 그릴 수 있는가? -> `matplotlib`의 `Path`와 `Ellipse` 사용

처음에 어렵긴 해도 example 몇 개만 보다보면 그릴만 했다. 이 글을 보시는 분들에게 많은 도움이 되었으면 좋겠다.
