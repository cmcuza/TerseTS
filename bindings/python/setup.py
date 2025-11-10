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


def copy_src_if_repository():
    """Copies the Zig source code to the current directory if in the repository."""
    cwd = Path.cwd()

    target_src_folder = cwd / "src"
    if target_src_folder.exists():
        return

    repository_root = cwd.parent.parent
    git_folder = repository_root / ".git"
    if git_folder.exists():
        input_src_folder = repository_root / "src"
        shutil.copytree(input_src_folder, target_src_folder)
        return

    raise FileNotFoundError("Failed to locate Zig source code.")


def delete_src_if_repository():
    """Deletes the Zig source code in the current directory if in the repository."""
    cwd = Path.cwd()

    src_folder = cwd / "src"
    if not src_folder.exists():
        return

    repository_root = cwd.parent.parent
    git_folder = repository_root / ".git"
    if git_folder.exists():
        shutil.rmtree(src_folder)


class ZigSDist(sdist):
    def run(self):
        copy_src_if_repository()
        super().run()
        delete_src_if_repository()


class ZigBDistWheel(bdist_wheel):
    def run(self):
        copy_src_if_repository()
        super().run()
        delete_src_if_repository()

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

        self.spawn(
            [
                sys.executable,
                "-m",
                "ziglang",
                "build-lib",
                ext.sources[0],
                f"-femit-bin={self.get_ext_fullpath(ext.name)}",
                "-mcpu",
                "native",
                "-dynamic",
                "-O",
                "ReleaseFast",
            ],
        )

        # Zig generates files that are not needed and can be removed.
        if sys.platform == "darwin" or sys.platform == "linux":
            os.remove(self.get_ext_fullpath(ext.name) + ".o")
        elif sys.platform == "win32":
            for name in ["capi.lib", "tersets.pdb", "tersets.pyd.obj"]:
                os.remove(os.path.join(self.build_lib, name))

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
