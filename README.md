# RTM Wrapper

[![Code style: black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

Common wrapper interface for [6S](https://salsa.umd.edu/6spage.html) and (eventually) [MODTRAN](http://modtran.spectral.com/).

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

## Misc References

* [Py6S by Robin Wilson](https://www.py6s.rtwilson.com/index.html)
* [6S | salsa.umd.edu](https://salsa.umd.edu/6spage.html)
