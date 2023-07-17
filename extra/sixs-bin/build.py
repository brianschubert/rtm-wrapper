"""
Poetry build script.

Downloads and builds 6S in a temporary directory, and it installs as a package
resource.

https://py6s.readthedocs.io/en/latest/installation.html#installing-6s
"""
import difflib
import hashlib
import io
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import textwrap

import urllib.request
import urllib.parse
from typing import Final

# Build script is always invoked from the project root directory.
_PACKAGE_ROOT: Final = pathlib.Path.cwd()

_SIXS_URL: Final = "https://rtwilson.com/downloads/6SV-1.1.tar"

_SIXS_NAME: Final = pathlib.PurePath(urllib.parse.urlparse(_SIXS_URL).path).name

_SIXS_SHA256: Final = "eedf652e6743b3991b5b9e586da2f55c73f9c9148335a193396bf3893c2bc88f"


class BuildError(RuntimeError):
    pass


def _is_windows() -> bool:
    return sys.platform == "win32"


def _download_sixs(directory: pathlib.Path) -> None:
    develop_sixs_cache = _PACKAGE_ROOT.joinpath(_SIXS_NAME)
    if develop_sixs_cache.exists():
        sixs_archive: bytes = develop_sixs_cache.read_bytes()
    else:
        sixs_archive: bytes = urllib.request.urlopen(_SIXS_URL).read()

    digest = hashlib.sha256(sixs_archive).hexdigest()
    if digest != _SIXS_SHA256:
        raise RuntimeError(
            f"6S archive hash validation valid. "
            f"Expected SHA256={_SIXS_SHA256}, got SHA256={digest}"
        )

    buffer = io.BytesIO(sixs_archive)
    tar_file = tarfile.open(fileobj=buffer, mode="r:")
    tar_file.extractall(directory)


def _assert_detect_command(cmd: list[str]) -> None:
    prog = cmd[0]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as ex:
        raise BuildError(f"unable to run {prog}") from ex
    print(f"detected {prog}:\n{textwrap.indent(result.stdout, '.. ')}", end="")


def build() -> None:
    with tempfile.TemporaryDirectory() as build_dir:
        build_dir = pathlib.Path(build_dir)

        print(f"Downloading 6S archive from '{_SIXS_URL}' to '{build_dir}'")
        _download_sixs(build_dir)
        for p in sorted(build_dir.glob("**/*")):
            print(f"file: {p}")

        # Check system dependencies
        _assert_detect_command(["make", "--version"])
        _assert_detect_command(["gfortran", "--version"])

        print("Building...")
        try:
            unix_arg = "FC=gfortran -std=legacy -ffixed-line-length-none -ffpe-summary=none $(FFLAGS)"
            subprocess.run(
                [
                    "make",
                    "sixs",
                    unix_arg if not _is_windows() else "",
                ],
                cwd=build_dir.joinpath("6SV1.1"),
                capture_output=True,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as ex:
            stdout = textwrap.indent(ex.stdout, ".. ")
            stderr = textwrap.indent(ex.stderr, ".. ")
            raise BuildError(
                f"failed to compile 6s.\nstdout:\n{stdout}\nstderr:\n{stderr}"
            ) from ex

        bin = build_dir.joinpath("6SV1.1").joinpath("sixsV1.1")

        print("Testing...")
        for example_in in build_dir.joinpath("Examples").glob("Example_In_*.txt"):
            example_out = example_in.parent.joinpath(
                example_in.name.replace("In", "Out")
            )
            try:
                result = subprocess.run(
                    [
                        bin,
                    ],
                    input=example_in.read_text(),
                    capture_output=True,
                    check=True,
                    text=True,
                    cwd=build_dir,
                )
            except subprocess.CalledProcessError as ex:
                stdout = textwrap.indent(ex.stdout, ".. ")
                stderr = textwrap.indent(ex.stderr, ".. ")
                raise BuildError(
                    f"running example {example_in.name} failed\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{stderr}"
                ) from ex
            expected = example_out.read_text()
            if result.stdout.strip() != expected.strip():
                diff = difflib.context_diff(
                    result.stdout.splitlines(),
                    expected.splitlines(),
                    fromfile="expected",
                    tofile="actual",
                )
                diff_text = "\n".join(diff)

                print(
                    f"WARNING: incorrect output for test {example_in.name}\n"
                    f"diff:\n{diff_text}"
                    # f"expected:\n{textwrap.indent(expected, '.. ')}\n"
                    # f"actual:\n{textwrap.indent(result.stdout, '.. ')}"
                )

        # Install 6S executable into package source.
        install_dest = _PACKAGE_ROOT / "src" / "sixs_bin" / bin.name
        shutil.copyfile(bin, install_dest)
        # Make sure file has owner execute permissions.
        os.chmod(install_dest, install_dest.stat().st_mode | stat.S_IXUSR)


if __name__ == "__main__":
    build()
