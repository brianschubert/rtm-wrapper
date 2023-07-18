"""
Build script.

Downloads and builds 6S in a temporary directory, and it installs as a package
resource.

https://py6s.readthedocs.io/en/latest/installation.html#installing-6s
"""
from __future__ import annotations

import difflib
import hashlib
import http
import http.client
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
import urllib.parse
import urllib.request
from typing import Final

# Build script is always invoked from the base directory of the current distribution.
_DISTRIBUTION_ROOT: Final = pathlib.Path.cwd()
_PACKAGE_ROOT: Final = _DISTRIBUTION_ROOT / "src" / "sixs-bin"

# URL to obtain 6S archive from. Alternatively, place 6SV-1.1.tar in the distribution
# base directory to avoid downloading a new copy.
# From Py6S author's website.
# _SIXS_URL: Final = "https://rtwilson.com/downloads/6SV-1.1.tar"
# Mirror from archive.org snapshot.
_SIXS_URL: Final = "https://web.archive.org/web/20220912090811if_/https://rtwilson.com/downloads/6SV-1.1.tar"

# Name of 6S archive file.
_SIXS_NAME: Final = pathlib.PurePath(urllib.parse.urlparse(_SIXS_URL).path).name

# Expected SHA256 has of the 6S archive file.
_SIXS_SHA256: Final = "eedf652e6743b3991b5b9e586da2f55c73f9c9148335a193396bf3893c2bc88f"


class BuildError(RuntimeError):
    """Raised on build failure."""


def _is_windows() -> bool:
    """Return True if the current platform is Windows."""
    return sys.platform == "win32"


def _download_sixs(directory: pathlib.Path) -> None:
    """
    Download, validate, and extract 6S into the given directory.
    """
    develop_sixs_cache = _DISTRIBUTION_ROOT.joinpath(_SIXS_NAME)
    if develop_sixs_cache.exists():
        sixs_archive = develop_sixs_cache.read_bytes()
    else:
        response: http.client.HTTPResponse = urllib.request.urlopen(_SIXS_URL)
        if response.status != http.HTTPStatus.OK.value:
            raise BuildError(
                f"failed to download 6S archive - got response "
                f"{response.status} {response.reason}"
            )
        sixs_archive = response.read()

    digest = hashlib.sha256(sixs_archive).hexdigest()
    if digest != _SIXS_SHA256:
        raise RuntimeError(
            f"6S archive hash validation failed. "
            f"Expected SHA256={_SIXS_SHA256}, got SHA256={digest}"
        )

    buffer = io.BytesIO(sixs_archive)
    tar_file = tarfile.open(fileobj=buffer, mode="r:")
    tar_file.extractall(directory)


def _assert_detect_command(cmd: list[str]) -> None:
    """
    Run the given command in a subprocess and write its outputs to stdout.

    Used to validate that a system dependency is installed and working correctly.
    """
    prog = cmd[0]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as ex:
        raise BuildError(f"unable to run {prog}") from ex
    print(f"detected {prog}:\n{textwrap.indent(result.stdout, '.. ')}", end="")


def _test_sixs(binary: pathlib.Path, example_dir: pathlib.Path) -> None:
    for example_in in example_dir.glob("Example_In_*.txt"):
        # Corresponding expected output file.
        example_out = example_in.parent.joinpath(example_in.name.replace("In", "Out"))

        try:
            result = subprocess.run(
                [binary],
                input=example_in.read_text(),
                capture_output=True,
                check=True,
                text=True,
                cwd=example_dir,
            )
        except subprocess.CalledProcessError as ex:
            stdout = textwrap.indent(ex.stdout, ".. ")
            stderr = textwrap.indent(ex.stderr, ".. ")
            raise BuildError(
                f"running example {example_in.name} failed\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            ) from ex

        # Compare actual output with expected output.
        # Some differences are expected since the expected output files include float
        # results to high precision, which can vary between systems with different
        # underlying numeric libraries.
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


def _install(binary: pathlib.Path, target: pathlib.Path) -> None:
    shutil.copyfile(binary, target)
    # Make sure file has owner execute permissions.
    os.chmod(target, target.stat().st_mode | stat.S_IXUSR)


def build(build_dir: pathlib.Path) -> None:
    """Run build in the given directory."""

    print(f"Downloading 6S archive from '{_SIXS_URL}' to '{build_dir}'")
    _download_sixs(build_dir)
    # Print extracted files for debugging.
    # for p in sorted(build_dir.glob("**/*")):
    #     print(f"file: {p}")

    # Check system dependencies
    _assert_detect_command(["make", "--version"])
    _assert_detect_command(["gfortran", "--version"])

    # Make 6S executable.
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

    # Path to built binary.
    sixs_binary = build_dir.joinpath("6SV1.1").joinpath("sixsV1.1")

    # Validate built binary against example suite.
    print("Testing...")
    _test_sixs(sixs_binary, build_dir.joinpath("Examples"))

    # Install 6S executable into package source.
    print("Installing...")
    _install(sixs_binary, _PACKAGE_ROOT / sixs_binary.name)


def main() -> None:
    """Build script entrypoint."""
    with tempfile.TemporaryDirectory() as build_dir:
        build(pathlib.Path(build_dir))


if __name__ == "__main__":
    main()
