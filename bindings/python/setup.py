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
import subprocess

from setuptools import setup
from setuptools.command.sdist import sdist
from setuptools.command.build_ext import build_ext


class ZigSDistExt(sdist):
    def run(self):
        shutil.copytree("../../src", "src")
        shutil.copyfile("../../build.zig", "build.zig")
        super().run()
        shutil.rmtree("src")
        os.remove("build.zig")


class ZigBuildExt(build_ext):
    def run(self):
        subprocess.check_call(
            [sys.executable, "-m", "ziglang", "build", "-Doptimize=ReleaseFast"],
            cwd=os.path.join(os.path.dirname(__file__), "src"),
        )
        super().run()


setup(
    cmdclass={"sdist": ZigSDistExt, "build_ext": ZigBuildExt},
)
