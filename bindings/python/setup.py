# Copyright 2024 TerseTS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from subprocess import CalledProcessError

from setuptools import Extension, find_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist


def copy_if_repository(relative_path):
    """Copies a file or directory at `relative_path` to the current directory if in the repository."""
    cwd = Path.cwd()
    target = cwd / relative_path
    if target.exists():
        return

    repository_root = cwd.parent.parent
    if not (repository_root / ".git").exists():
        raise FileNotFoundError(f"Failed to locate {relative_path}.")

    source = repository_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def find_zig_command():
    """Find the command required for running the zig compiler. It tries the
    official ziglang PyPi package first, and then tries zig in PATH."""

    # zig_version is extracted using a regex as tomllib is not in Python 3.7
    # and the format for specifying a specific version is not likely to change.
    pyproject = Path("pyproject.toml").read_text()
    zig_version = re.search(r'"ziglang\s*==\s*([^";]+)', pyproject).group(1).strip()

    # Determine if there is a version of Zig that can be used to compile the package.
    for zig_cmd in [[sys.executable, "-m", "ziglang"], ["zig"]]:
        try:
            zig_version_number = subprocess.run(zig_cmd + ["version"], capture_output=True, text=True, check=True)
        # A CalledProcessError exception is raised if a process does not complete
        # successfully, and a OSError exception is raised if the binary is not found.
        except (CalledProcessError, OSError):
            continue

        if zig_version_number.stdout.strip() == zig_version:
            return zig_cmd

    raise RuntimeError(
        f"Failed to locate Zig {zig_version}, install ziglang {zig_version} from PyPI or add zig {zig_version} to PATH."
    )


def delete_if_repository(relative_path):
    """Deletes a file or directory at `relative_path` in the current directory if in the repository."""
    cwd = Path.cwd()
    target = cwd / relative_path
    if not target.exists():
        return

    if not (cwd.parent.parent / ".git").exists():
        return

    if target.is_dir():
        shutil.rmtree(target)
        parent = target.parent
        if parent != cwd and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    else:
        target.unlink()


class ZigSDist(sdist):
    def run(self):
        copy_if_repository("src")
        copy_if_repository("lib/pocketfft")
        copy_if_repository("build.zig")
        try:
            super().run()
        finally:
            delete_if_repository("src")
            delete_if_repository("lib/pocketfft")
            delete_if_repository("build.zig")


class ZigBDistWheel(bdist_wheel):
    def run(self):
        copy_if_repository("src")
        copy_if_repository("lib/pocketfft")
        copy_if_repository("build.zig")
        try:
            super().run()
        finally:
            delete_if_repository("src")
            delete_if_repository("lib/pocketfft")
            delete_if_repository("build.zig")

    def get_tag(self):
        python, abi, plat = super().get_tag()
        # TerseTS is a native library, not a Python extension.
        python, abi = "py3", "none"
        return python, abi, plat


class ZigBuildExt(build_ext):
    def build_extension(self, ext):
        assert len(ext.sources) == 1

        # Zig requires that the directories exists.
        if not os.path.exists(self.build_lib):
            os.makedirs(self.build_lib)

        # Output path for the generated library.
        output_path = self.get_ext_fullpath(ext.name)

        optimize = "Debug" if self.debug else "ReleaseFast"
        zig_cmd = find_zig_command()
        zig_cmd.extend(
            [
                "build",
                "-Dlinking=dynamic",
                f"-Doptimize={optimize}",
            ]
        )
        self.spawn(zig_cmd)

        if sys.platform == "win32":
            built_library = Path("zig-out") / "bin" / "tersets.dll"
        elif sys.platform == "darwin":
            built_library = Path("zig-out") / "lib" / "libtersets.dylib"
        else:
            built_library = Path("zig-out") / "lib" / "libtersets.so"

        if not built_library.exists():
            raise FileNotFoundError(f"Expected Zig output was not found: {built_library}")

        shutil.copy2(built_library, output_path)

    def get_ext_filename(self, ext_name):
        # Removes the CPython part of ext_name as the library is not linked to
        # CPython and it simplifies the code for loading the library in Python.
        filename = super().get_ext_filename(ext_name)
        start = filename.find(".")
        end = filename.rfind(".")
        return filename[:start] + filename[end:]


setup(
    packages=find_packages(exclude=("tests")),
    ext_modules=[Extension("tersets", sources=["src/capi.zig"])],
    cmdclass={
        "sdist": ZigSDist,
        "bdist_wheel": ZigBDistWheel,
        "build_ext": ZigBuildExt,
    },
)
