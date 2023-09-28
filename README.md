# RTM Wrapper

[![Code style: black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

Common wrapper interface for various Radiative Transfer Models (RTMs).

Currently supports [6S](https://salsa.umd.edu/6spage.html).


## Install

With existing 6S installation:
```shell
$ pip install .
```

With locally compiled 6S binary:
```shell
$ pip install '.[6s]'
```

Without downloading dependencies:
```shell
$ pip install --no-deps .
```

## Test
```
$ pytest
```
