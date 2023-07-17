# 6S Binary Wrapper

Convenience package for building and installing a local 6S v1.1 executable.

Install this package by enabling the `6s` feature in the main package.

## Testing

```pycon
>>> import sixs_bin
>>> sixs_bin.test_wrapper()
6S wrapper script by Robin Wilson
Using 6S located at /path/to/venv/lib/python3.X/site-packages/sixs_bin/sixsV1.1
Running 6S using a set of test parameters
6sV version: 1.1
The results are:
Expected result: 619.158000
Actual result: 619.158000
#### Results agree, Py6S is working correctly
```
