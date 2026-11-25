# ruff: noqa: D101, D102
"""Tests for locating the STEER cost model folder (``solve_network.resolve_steer_dir``).

Each test builds a fake project in a temporary folder::

    <root>/pypsa-usa/workflow/scripts/solve_network.py   (the script path passed in)
    <root>/02_STEERMODEL/  and/or  <root>/openlearning/

so ``parents[3]`` of the script path is ``<root>``, as in the real layout.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from solve_network import add_steer_to_path, resolve_steer_dir


def _project(tmp_path, *folders):
    script = tmp_path / "pypsa-usa" / "workflow" / "scripts" / "solve_network.py"
    script.parent.mkdir(parents=True)
    script.touch()
    for name in folders:
        (tmp_path / name).mkdir()
    return script


class TestResolveSteerDir:
    def test_existing_layout_resolves_as_before(self, tmp_path):
        """No config key and a 02_STEERMODEL folder: the same path the old code built."""
        script = _project(tmp_path, "02_STEERMODEL")
        got = resolve_steer_dir({}, script_path=script)
        assert got == Path(script).resolve().parents[3] / "02_STEERMODEL"

    def test_02_steermodel_preferred_over_openlearning(self, tmp_path):
        script = _project(tmp_path, "02_STEERMODEL", "openlearning")
        assert resolve_steer_dir({"costs": {}}, script_path=script).name == "02_STEERMODEL"

    def test_openlearning_used_when_02_steermodel_absent(self, tmp_path):
        script = _project(tmp_path, "openlearning")
        assert resolve_steer_dir({}, script_path=script) == tmp_path.resolve() / "openlearning"

    def test_nothing_found_raises_listing_all_tried(self, tmp_path):
        script = _project(tmp_path)
        with pytest.raises(FileNotFoundError, match=r"02_STEERMODEL.*openlearning.*steer_model_dir"):
            resolve_steer_dir({}, script_path=script)

    def test_config_absolute_path_wins(self, tmp_path):
        script = _project(tmp_path, "02_STEERMODEL")
        elsewhere = tmp_path / "somewhere" / "steer_model"
        elsewhere.mkdir(parents=True)
        cfg = {"costs": {"steer_model_dir": str(elsewhere)}}
        assert resolve_steer_dir(cfg, script_path=script) == elsewhere

    def test_config_relative_path_is_relative_to_project_root(self, tmp_path):
        script = _project(tmp_path, "02_STEERMODEL", "my_steer")
        cfg = {"costs": {"steer_model_dir": "my_steer"}}
        assert resolve_steer_dir(cfg, script_path=script) == tmp_path.resolve() / "my_steer"

    def test_config_missing_folder_raises_even_if_default_exists(self, tmp_path):
        script = _project(tmp_path, "02_STEERMODEL")
        cfg = {"costs": {"steer_model_dir": "does_not_exist"}}
        with pytest.raises(FileNotFoundError, match=r"steer_model_dir.*does_not_exist"):
            resolve_steer_dir(cfg, script_path=script)

    def test_config_path_to_a_file_raises(self, tmp_path):
        script = _project(tmp_path)
        (tmp_path / "a_file").touch()
        with pytest.raises(FileNotFoundError, match="not an existing folder"):
            resolve_steer_dir({"costs": {"steer_model_dir": "a_file"}}, script_path=script)


class TestAddSteerToPath:
    @pytest.fixture(autouse=True)
    def clean_state(self, monkeypatch):
        """Work on a copy of sys.path and with no steer module imported yet."""
        monkeypatch.setattr(sys, "path", list(sys.path))
        for name in [m for m in sys.modules if m == "steer" or m.startswith("steer.")]:
            monkeypatch.delitem(sys.modules, name)

    def _steer_copy(self, folder, regular=True):
        """Create folder/steer, a regular package (with __init__.py) or a namespace one."""
        pkg = folder / "steer"
        pkg.mkdir(parents=True)
        if regular:
            (pkg / "__init__.py").write_text("")
        return folder

    def _imported_steer_dir(self):
        importlib.invalidate_caches()
        steer = importlib.import_module("steer")
        return Path(next(iter(steer.__path__))).resolve().parent

    def test_added_at_front_when_steer_not_importable(self, tmp_path):
        add_steer_to_path(tmp_path)
        assert sys.path[0] == str(tmp_path)

    def test_folder_goes_in_front_even_when_steer_installed(self, tmp_path):
        """A steer reachable elsewhere on the path (as if pip-installed) does not win."""
        installed = self._steer_copy(tmp_path / "site-packages")
        local = self._steer_copy(tmp_path / "02_STEERMODEL")
        sys.path.insert(0, str(installed))
        add_steer_to_path(local)
        assert sys.path[0] == str(local)
        assert self._imported_steer_dir() == local.resolve()

    def test_namespace_package_case_still_inserts(self, tmp_path):
        """A bare steer folder without __init__.py elsewhere on the path does not win."""
        bare = self._steer_copy(tmp_path / "elsewhere", regular=False)
        local = self._steer_copy(tmp_path / "02_STEERMODEL")
        sys.path.insert(0, str(bare))
        add_steer_to_path(local)
        assert sys.path[0] == str(local)
        assert self._imported_steer_dir() == local.resolve()

    def test_folder_later_in_path_moved_to_front_without_duplicate(self, tmp_path):
        sys.path.append(str(tmp_path))
        add_steer_to_path(tmp_path)
        assert sys.path[0] == str(tmp_path)
        assert sys.path.count(str(tmp_path)) == 1

    def test_not_added_twice(self, tmp_path):
        add_steer_to_path(tmp_path)
        add_steer_to_path(tmp_path)
        assert sys.path[0] == str(tmp_path)
        assert sys.path.count(str(tmp_path)) == 1

    def test_steer_already_imported_from_elsewhere_raises(self, tmp_path):
        other = self._steer_copy(tmp_path / "site-packages")
        local = self._steer_copy(tmp_path / "02_STEERMODEL")
        sys.path.insert(0, str(other))
        self._imported_steer_dir()
        with pytest.raises(ImportError, match=r"site-packages.*02_STEERMODEL"):
            add_steer_to_path(local)

    def test_steer_already_imported_from_resolved_folder_does_not_raise(self, tmp_path):
        local = self._steer_copy(tmp_path / "02_STEERMODEL")
        sys.path.append(str(local))
        assert self._imported_steer_dir() == local.resolve()
        add_steer_to_path(local)
        assert sys.path[0] == str(local)
        assert sys.path.count(str(local)) == 1
