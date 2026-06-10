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
import sys
import shutil
from pathlib import Path

from setuptools import setup, find_packages, Extension
from setuptools.command.sdist import sdist
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_wheel import bdist_wheel


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
        zig_cmd = [
            sys.executable,
            "-m",
            "ziglang",
            "build",
            "-Dlinking=dynamic",
            f"-Doptimize={optimize}",
        ]
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
