"""Julia bindings for the TerseTS library."""

# Copyright 2026 TerseTS Contributors
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

module TerseTS

function __init__()
    if  Sys.isapple()
        print("Hello Mac")
    elseif Sys.isunix()
        print("Hello Unix")
    elseif Sys.iswindows()
        print("Hello Windowos")
    else
        error("Could not find TerseTS: looked '*{library_name}' in {library_folder}")
    end
end

end
