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


def copy_pocketfft_if_repository():
    """Copies PocketFFT C source code to the current directory if in the repository."""
    cwd = Path.cwd()

    target_pocketfft_folder = cwd / "lib" / "pocketfft"
    if target_pocketfft_folder.exists():
        return False

    repository_root = cwd.parent.parent
    git_folder = repository_root / ".git"
    if git_folder.exists():
        input_pocketfft_folder = repository_root / "lib" / "pocketfft"
        target_pocketfft_folder.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(input_pocketfft_folder, target_pocketfft_folder)
        return 

    raise FileNotFoundError("Failed to locate PocketFFT source code.")


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


def delete_pocketfft_if_repository():
    """Deletes the PocketFFT source code in the current directory if in the repository."""
    cwd = Path.cwd()

    pocketfft_folder = cwd / "lib" / "pocketfft"
    if not pocketfft_folder.exists():
        return

    repository_root = cwd.parent.parent
    git_folder = repository_root / ".git"
    if git_folder.exists():
        shutil.rmtree(pocketfft_folder)
        lib_folder = cwd / "lib"
        if lib_folder.exists() and not any(lib_folder.iterdir()):
            lib_folder.rmdir()


class ZigSDist(sdist):
    def run(self):
        copy_src_if_repository()
        copy_pocketfft_if_repository()
        try:
            super().run()
        finally:
            delete_src_if_repository()
            delete_pocketfft_if_repository()


class ZigBDistWheel(bdist_wheel):
    def run(self):
        copy_src_if_repository()
        copy_pocketfft_if_repository()
        try:
            super().run()
        finally:
            delete_src_if_repository()
            delete_pocketfft_if_repository()

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
        
        # Zig's build-lib command.
        zig_cmd = [sys.executable, "-m", "ziglang", "build-lib"]
        
        # First source file, then cflags, then additional C source file for PocketFFT.
        zig_inputs = [
            ext.sources[0],
            "-cflags",
            "-std=c99",
            "--",
            "lib/pocketfft/pocketfft.c",
        ]
        # Zig's build-lib flags.
        zig_flags = [
            f"-femit-bin={output_path}",
            "-Ilib/pocketfft",
            "-mcpu",
            "native",
            "-dynamic",
            "-lc",
            "-O",
            "ReleaseFast",
        ]

        self.spawn(zig_cmd + zig_inputs + zig_flags)

        # Zig generates files that are not needed and can be removed.
        if sys.platform == "darwin" or sys.platform == "linux":
            path_to_file = self.get_ext_fullpath(ext.name) + ".o"
            if os.path.exists(path_to_file):
                os.remove(path_to_file)

        elif sys.platform == "win32":
            for name in ["capi.lib", "tersets.pdb", "tersets.pyd.obj"]:
                path_to_file = os.path.join(self.build_lib, name)
                if os.path.exists(path_to_file):
                    os.remove(path_to_file)

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
