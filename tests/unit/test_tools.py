# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from app.agent import get_current_time, get_weather


def test_get_current_time_valid_city() -> None:
    res = get_current_time("San Francisco")
    assert "The current time for query San Francisco is" in res


def test_get_current_time_unsupported_city_returns_recovery_instructions() -> None:
    res = get_current_time("Tokyo")
    assert "Error: Timezone information is unavailable" in res
    assert "Please ask the user to clarify or specify a supported location" in res


def test_get_weather() -> None:
    assert "foggy" in get_weather("SF")
    assert "sunny" in get_weather("New York")
