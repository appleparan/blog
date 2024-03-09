---
layout: post
title: Multiple Plots with Map in Python
author: jongsukim
date: 2021-08-15 12:00:00 +0900
categories: [Programming, Python]
tags:
  - GeoPandas
  - Matplotlib
  - Data Visualization
math: true
---

# Introduction

[이전 포스트](https://blog.liam.kim/2021/08/14/Plot-Geospatial-Data-In-Python/)에서는
두 지도를 한 Figure에 그리는 것을 설명했는데, 이번에는 한 지도에서 특정 포인트마다 여러 개의 Plot을 어떻게 그리는지 설명하고자 한다.

Plot의 조건은 다음과 같다.

* 종로구, 서초구, 강서구 이 세 지점에 각각 세 가지의 plot를 그릴 예정이다. (총 9개)
* 각 지점에 그리는 plot를 A, B, C라고 한다.
* A plot는 선형(linear, \\(y=ax\\))의 plot를 그린다.
* B plot는 이차 함수(quadratic, \\(y=ax^2\\))의 plot를 그린다.
* C plot는 삼각 함수(trigonometric, \\(y=a\sin{bx}\\))의 plot를 그린다.
* \\(a\\)와 \\(b\\)는 상수이다.
* 비교를 위해서 A, B, C plot의 y축은 각각 plot별로 min, max를 통일시켜야한다.

이를 위해서 다음과 같은 절차를 밟는다.

1. 서울의 지도를 그린다.
2. 종로구, 서초구, 강서구 이 세 지점(station)에 원을 그린다.
3. 세 지점의 원을 가리지 않으면서 근처라고 할 수 있는 곳에 사각형을 그린다.
4. Zoom되었다는 효과를 넣기 위해 사각형과 원을 선으로 잇는다.
5. 사각형안에 `matplotlib`의 `inset` 3개를 지정한다.
6. 각각의 `matplotlib`의 `inset`에 A, B, C plot를 그린다.
7. 각각의 `inset`에 A, B, C로 표시되는 annotation을 삽입한다.

여기에 나오는 코드는 이전과 마찬가지로 [Colab](https://colab.research.google.com/drive/1Xx2LIHmQ4TxkZ3yhjitsC_O6eobv8xPw?usp=sharing)에 공개한다.

# Draw Simple Seoul Map

[이전 포스트](https://blog.liam.kim/2021/08/14/Plot-Geospatial-Data-In-Python/)에서 썼던 서울의 지도 데이터를 이용하여 그림을 그린다.

```Python
# download seoul geojson data
seoul_url = 'https://github.com/southkorea/seoul-maps/raw/master/kostat/2013/json/seoul_municipalities_geo_simple.json'
seoul_df = gpd.read_file(seoul_url)
seoul_df.plot(figsize=(7.22, 7.22),
                  color='none', edgecolor='#333', facecolor='none', alpha=0.3)
fig = plt.gcf()
```

![Seoul Map](/assets/images/post/2021-08-15-Multiple-Plots-in-Map/Seoul-station.svg)

# Draw Circle in Station Location
## Station Information

각각의 station 정보를 따로 저장한다. 각 station의 이름을 key로 하는 dictionary로 조금이나마 중복된 코드를 줄이고자 하였다. 각 파라미터는 try-and-error로 정해진 하드 코딩되는 값이다.

```Python
stations_map = {
    '종로구': {
        'lat': 127.0050,
        'lon': 37.5720,
        'box_left': 0.45,
        'box_bottom': 0.72,
        'annot_x': 0.001,
        'annot_y': -0.012,
        'eng_name': 'Jongno',
        'loc': 'lower left',
        'loc0': 3,
        'loc1': 4,
        'posx': 20,
        'posy': 20,
        'a': 1.0,
        'b': 1.0},
    '강서구': {
        'lat': 126.8351,
        'lon': 37.5447,
        'box_left': 0.01,
        'box_bottom': 0.48,
        'annot_x': 0.008,
        'annot_y': -0.008,
        'eng_name': 'Gangseo',
        'loc': 'lower left',
        'loc0': 3,
        'loc1': 4,
        'posx': 0,
        'posy': -20,
        'a': 2.0,
        'b': 2.0},
    '서초구': {
        'lat': 126.9945,
        'lon': 37.5046,
        'box_left': 0.42,
        'box_bottom': 0.04,
        'annot_x': 0.005,
        'annot_y': 0.004,
        'eng_name': 'Seocho',
        'loc': 'lower left',
        'loc0': 1,
        'loc1': 2,
        'posx': 0,
        'posy': -20,
        'a': 3.0,
        'b': 3.0}
}
```

## Plot Circle for Stations

각각의 station의 위치를 표시하는 원을 그린다.

```Python
for station_name in stations_map.keys():
    lat = stations_map[station_name]['lat']
    lon = stations_map[station_name]['lon']

    aspect = ax.get_aspect()
    point_r = 0.008
    p = Ellipse((lat, lon), point_r, point_r / aspect,  zorder=6)
    ax.add_artist(p)
```

![Station Circles in Seoul Map](/assets/images/post/2021-08-15-Multiple-Plots-in-Map/Seoul-station-circle.svg)

# Draw Rectangles

각 plot의 너비와 높이를 `w`와 `h`라고 하자.
`w`와 `h`의 크기는 Axes의 상대적인 크기를 사용하기로 하였다. 따라서 `transform=ax.transAxes` 전달 인자를 사용한다.

각 Rectangle은 3개의 plot이 가로로 있는 형태를 그릴 예정이므로 기본적으로 `3*w + h`의 크기를 지닌다고 할 수 있다. 하지만, 좌표축 레이블(axis label)이나 tick의 존재 때문에 padding이 필요하다. 이를 `w_pad`와 `h_pad`라 하면 `w`의 30%, `h`의 15%로 지정하였다. 직사각형의 크기는 `3*w+3.8*w_pad`, `h+5.2*h_pad`로 지정하였다. 3.8와 5.2은 큰 의미가 있는 것은 아니고 plot를 실제로 그리고 조정하면서 try-and-error로 여백을 조정하다보니 그렇게 되었다.

이를 `Rectangle`을 이용하여 그린다. `station_map`의 `box_left`과 `box_bottom`은 `Rectangle`의 xy 즉, 왼쪽 아래 좌표를 나타낸다. 이 또한 try-and-error로 위에서 그렸던 원을 가리지 않으면서 서로 겹치지도 않는 적절한 위치를 찾아서 조정하였다.

다음 그림을 위해 `Rectangle object`는 `rects`라는 `dictionary`에 따로 저장하였다.

```Python
# set size of rectangle according to Axes coordinate
w, h = 0.13, 0.13
w_pad, h_pad = w*0.3, h*0.15

rect_w = 3*w + 3.8*w_pad
rect_h = h + 5.2*h_pad
rects = {}

for station_name in stations_map.keys():
    rect = Rectangle((stations_map[station_name]['box_left'],
        stations_map[station_name]['box_bottom']),
        rect_w, rect_h, transform=ax.transAxes,
        linewidth=0.5, edgecolor='k', facecolor='white', zorder=6)
    ax.add_artist(rect)
    rects[station_name] = rect
```

![Station Circles and Rectangles in Seoul Map](/assets/images/post/2021-08-15-Multiple-Plots-in-Map/Seoul-station-rect.svg)

# Draw Line between Circle and Rectangle

Station의 정보를 확대해서 보여준다는 의미로 선을 그릴 필요가 있다. [이전 포스트](https://blog.liam.kim/2021/08/14/Plot-Geospatial-Data-In-Python/)에서 설명한 것과 같이 선을 그리면 된다.

여기서 중요한 것은 좌표축인데, circle의 위치를 알려주는 `px`, `py`는 위도와 경도로 된, 즉 데이터에 기반한 좌표이다 (`ax.transData`). 반면에, `Rectangle`은 `matplotlib`의 `Axes`의 상대적인 크기에 기반한 좌표이다 (`ax.transAxes`). 따라서 이 둘을 통일할 필요가 있다. 이는 [`Transform`의 `inverted`](https://matplotlib.org/stable/api/transformations.html#matplotlib.transforms.Transform.inverted) method와 [transformation pipeline](https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html#the-transformation-pipeline)을 이용하면 해결할 수 있다. 이를 정리하면 다음 코드이다.

```
axis_to_data = ax.transAxes + ax.transData.inverted()
x0, y0 = axis_to_data.transform(rects[station_name].xy)
x1, y1 = axis_to_data.transform((rects[station_name].xy[0] + rect_w, rects[station_name].xy[1] + rect_h))
px, py = lat, lon
```

또한 위에서 `Rectangle`을 그릴 때 `zorder=6`을 지정했는데, 이는 선이 plot을 가리지 않게 하기 위해서 지정하였다.

선을 그리는 전체 코드는 다음과 같다.

```Python
for station_name in stations_map.keys():
    lat = stations_map[station_name]['lat']
    lon = stations_map[station_name]['lon']

    # transformation pipeline
    axis_to_data = ax.transAxes + ax.transData.inverted()
    x0, y0 = axis_to_data.transform(rects[station_name].xy)
    x1, y1 = axis_to_data.transform((rects[station_name].xy[0] + rect_w, rects[station_name].xy[1] + rect_h))
    px, py = lat, lon
    if stations_map[station_name]['loc0'] == 1:
        # upper right
        verts_0 = [(px, py), (x1, y1), (px, py)]
    elif stations_map[station_name]['loc0'] == 2:
        # upper left
        verts_0 = [(px, py), (x0, y1), (px, py)]
    elif stations_map[station_name]['loc0'] == 3:
        # lower left
        verts_0 = [(px, py), (x0, y0), (px, py)]
    elif stations_map[station_name]['loc0'] == 4:
        # lower right
        verts_0 = [(px, py), (x1, y0), (px, py)]
    codes_0 = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]

    if stations_map[station_name]['loc1'] == 1:
        # upper right
        verts_1 = [(px, py), (x1, y1), (px, py)]
    elif stations_map[station_name]['loc1'] == 2:
        # upper left
        verts_1 = [(px, py), (x0, y1), (px, py)]
    elif stations_map[station_name]['loc1'] == 3:
        # lower left
        verts_1 = [(px, py), (x0, y0), (px, py)]
    elif stations_map[station_name]['loc1'] == 4:
        # lower right
        verts_1 = [(px, py), (x1, y0), (px, py)]
    codes_1 = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]

    path_0 = mpath.Path(verts_0, codes_0)
    path_1 = mpath.Path(verts_1, codes_1)

    patch_0 = ax.add_patch(mpatches.PathPatch(path_0, facecolor='k', lw=0.5))
    patch_1 = ax.add_patch(mpatches.PathPatch(path_1, facecolor='k', lw=0.5))
```

![Station Circles, Rectangles, and Lines in Seoul Map](/assets/images/post/2021-08-15-Multiple-Plots-in-Map/Seoul-station-line.svg)

# Plot in Insets

여러 개의 plot을 한 figure 안에 그려야하므로, 이번에도 inset을 사용한다. 여기가 이제 상당히 고통스러운 부분이다. 위에서 언급한 `w_pad`와 `h_pad`, 그리고 axis label와 ticklabel을 고려한 `w_offset`과 `h_offset`을 잘 조정해서 가장 적절한 여백값을 찾아야한다. tick의 숫자의 크기, 폰트의 크기 등에 따라 달라질 수 있으며 이는 plot를 같이 그려야 체크할 수 있기에 plot 또한 같이 그린다.

이 때, `zorder`는 default값인 5, 그리고 `Rectangle`에서 설정한 6보다 큰 7을 설정함으로써 다른 요소들에 의해 가려지지 않도록 한다.

그리고 조건대로 각각의 plot 유형마다 한계값(ylim)을 통일시킨다.

```Python
insets = {}
w_offset = 1.5 * w_pad
h_offset = 3 * h_pad
nx = 101

def style_insets(insets):
    for inset in insets:
        inset.set_title("")
        # show grid
        inset.xaxis.grid(True, visible=True, which='major')
        inset.yaxis.grid(True, visible=True, which='major')

        # small tick label
        for tick in inset.xaxis.get_major_ticks():
            tick.label.set_fontsize('xx-small')
        for tick in inset.yaxis.get_major_ticks():
            tick.label.set_fontsize('xx-small')

        # ticks are close to axis
        inset.tick_params(axis='x', which='major', pad=1)
        inset.tick_params(axis='y', which='major', pad=1)
        # x axis label is close to x axis
        inset.set_xlabel('x', fontsize='x-small', labelpad=1.5)

    # y axis label is shown in first plot only
    # y axis label is close to y axis
    insets[0].set_ylabel('y', fontsize='small', labelpad=0.5)

# initialize min/max by max/min of 'float'
tot_ylims = {
        0: [np.finfo('float').max, np.finfo('float').min],
        1: [np.finfo('float').max, np.finfo('float').min],
        2: [np.finfo('float').max, np.finfo('float').min]}

for station_name in stations_map.keys():
    axin0 = ax.inset_axes(bounds=[stations_map[station_name]['box_left'] + w_offset,
                                stations_map[station_name]['box_bottom'] + h_offset,
                                w, h], transform=ax.transAxes, zorder=7)
    axin1 = ax.inset_axes(bounds=[stations_map[station_name]['box_left'] + w + w_pad + w_offset,
                                stations_map[station_name]['box_bottom'] + h_offset,
                                w, h], transform=ax.transAxes, zorder=7)
    axin2 = ax.inset_axes(bounds=[stations_map[station_name]['box_left'] + 2*w + 2*w_pad + w_offset,
                                stations_map[station_name]['box_bottom'] + h_offset,
                                w, h], transform=ax.transAxes, zorder=7)
    # store inset for later use
    insets[station_name] = [axin0, axin1, axin2]

    # prepare data
    xs0 = np.linspace(-5.0, 5.0, num=nx, endpoint=True)
    xs1 = np.linspace(-3.0, 3.0, num=nx, endpoint=True)
    xs2 = np.linspace(-2.0*np.pi, 2.0*np.pi, num=nx, endpoint=True)
    ys0 = float(stations_map[station_name]['a']) * xs0
    ys1 = float(stations_map[station_name]['a']) * np.power(xs1, 2)
    ys2 = float(stations_map[station_name]['a']) * \
        np.sin(float(stations_map[station_name]['b'])*xs2)

    # plot to inset
    axin0.plot(xs0, ys0, color='k')
    axin1.plot(xs1, ys1, color='g')
    axin2.plot(xs2, ys2, color='r')

    # store min/max of ylim
    tot_ylims[0][0] = min(tot_ylims[0][0], axin0.get_ylim()[0])
    tot_ylims[1][0] = min(tot_ylims[1][0], axin1.get_ylim()[0])
    tot_ylims[2][0] = min(tot_ylims[2][0], axin2.get_ylim()[0])

    tot_ylims[0][1] = max(tot_ylims[0][1], axin0.get_ylim()[1])
    tot_ylims[1][1] = max(tot_ylims[1][1], axin1.get_ylim()[1])
    tot_ylims[2][1] = max(tot_ylims[2][1], axin2.get_ylim()[1])

    # customize style of inset
    style_insets([axin0, axin1, axin2])

# set same y limit per plot type
for station_name in stations_map.keys():
    insets[station_name][0].set_ylim(tot_ylims[0][0], tot_ylims[0][1])
    insets[station_name][1].set_ylim(tot_ylims[1][0], tot_ylims[1][1])
    insets[station_name][2].set_ylim(tot_ylims[2][0], tot_ylims[2][1])
```

![Multiple Plots per Station in Seoul Map](/assets/images/post/2021-08-15-Multiple-Plots-in-Map/Seoul-station-plot.svg)

# Annotation

Plot이 많은 경우에는 이를 인용하기 위해 Annotation이 필요하다. 각 Circle (station)에 대한 annotation, 그리고 각 plot의 유형(A, B, C)에 대한 annotation을 다음과 같은 코드로 구현한다.

Circle의 annotation 위치는 station 별로 따로 설정하고(`xycoords='data'`), 각 plot의 유형(A, B, C)에 대한 annotation은 각 inset에서의 상대적인 위치(`xycoords='axes fraction'`)에 대해 고정된 위치를 세팅한다 (`(-0.18, 1.08)`). 이 또한 plot하면서 조정해야하는 수치이다.

```
# slice alphabets by length of types
multipanel_labels = np.array(list(string.ascii_uppercase)[:3])

for (i, station_name) in enumerate(stations_map.keys()):
    lat = stations_map[station_name]['lat']
    lon = stations_map[station_name]['lon']
    # annotate station name on Axes
    ax.annotate(stations_map[station_name]['eng_name'], (lat + stations_map[station_name]['annot_x'],
                                lon + stations_map[station_name]['annot_y']),
                        xycoords='data',
                        fontsize='medium')
    # annotate type of plot on inset
    for ii in range(3):
        insets[station_name][ii].annotate(
            multipanel_labels[ii], (-0.18, 1.08),
            xycoords='axes fraction',
            fontsize='medium', fontweight='bold')
fig.tight_layout(pad=0.15)
```

![Multiple Plots per Station in Seoul Map](/assets/images/post/2021-08-15-Multiple-Plots-in-Map/Seoul-station-annot.svg)

# Conclusion

이걸 그리던 2021년 2월 당시에 상당히 고민해서 그린 거였는데, `Rectangle`과 `inset`을 계층적인 구조로 그리려고 시도했던게 복잡성을 키운 셈이 되어버려서 잘 그려지지 않았다. 따로따로 생각하고 약간의 try-and-error를 첨가하니 오히려 더 쉽게 그려져서 허망했던 기억이 난다. 다만, 6개월 사이에 [`subplot`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html)이라는게 생겼기 때문에 좀 더 쉽게 그릴 방법이 있지 않을까 생각한다.
