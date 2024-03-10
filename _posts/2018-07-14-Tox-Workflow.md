---
layout: post
title: (Note) Tox workflow
author: jongsukim
date: 2017-07-17 12:00:00 +0900
categories: [Programming, Python]
tags:
  - python
  - tox
math: false
---

## What do I need to install

* `tox` : virtualenv for testing
* `tox-pyenv` : `pyenv` plugin for `tox`
* `pytest` : testing framework
* `pipreqs` : generates `requirements.txt`

## What do I need to do

1. Checkout [`cookiecutter`](https://github.com/audreyr/cookiecutter)

2. create virtualenv using pyenv

```shell
$ pyenv install "some versions"
$ pip install tox tox-pyenv
$ pyenv local "my_env" "some versions"
```

3. create `src` directory and change some settings

   * Why I use `src/` directory?
     * Check Hynek's [post](https://hynek.me/articles/testing-packaging/)

1. As the post says, modify `setup.py`


## Example of executing tox

```shell
$ pyenv install 3.8.2
$ pyenv virtualenv -p Python3.8 3.8.2 my_env
$ pip install tox tox-pyenv
$ pyenv local my_env 3.8.2
```

  write code and tests then

```shell
$ pipreqs .
$ pip install -e .
$ tox
```


