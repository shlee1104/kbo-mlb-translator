"""Execute app.py end to end against a stubbed Streamlit.

`app.py` is layout, and layout is exactly where a column that does not
exist or a widget handed the wrong type hides: none of the unit tests ever
call it. Replacing `streamlit` with a recording stub runs every line of the
real file against the real data, which is how the pitcher page was found to
crash on open.

It cannot tell you the page looks right. It tells you the page renders,
which is the failure mode that actually happened.

Skipped when the pipeline has not been run, since there is nothing to
render without data.
"""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"


class FakeStreamlit:
    """Answers every widget with its default, and records nothing else.

    Returning the DEFAULT rather than a fixed value matters: the defaults
    are what a user sees on first load, so this exercises the configuration
    that ships.
    """

    def __init__(self, name="st", side="batting"):
        self._name = name
        self._side = side

    def __getattr__(self, item):
        if item == "sidebar":
            return FakeStreamlit("sidebar", self._side)
        return FakeStreamlit(f"{self._name}.{item}", self._side)

    def __call__(self, *args, **kwargs):
        kind = self._name.split(".")[-1]

        if kind == "radio":
            return self._side
        if kind == "selectbox":
            return list(args[1])[0]
        if kind == "select_slider":
            return kwargs.get("value", list(args[1])[0])
        if kind == "checkbox":
            return kwargs.get("value", True)
        if kind == "number_input":
            return args[3] if len(args) > 3 else 2026
        if kind == "slider":
            return args[3] if len(args) > 3 else args[1]
        if kind == "multiselect":
            return kwargs.get("default", args[1])
        if kind == "columns":
            n = args[0] if isinstance(args[0], int) else len(args[0])
            return [FakeStreamlit("col", self._side) for _ in range(n)]
        if kind == "tabs":
            return [FakeStreamlit("tab", self._side) for _ in args[0]]
        if kind == "expander":
            return FakeStreamlit("expander", self._side)
        if kind == "cache_data":
            if args and callable(args[0]):
                return args[0]
            return lambda fn: fn
        if kind == "stop":
            raise SystemExit("st.stop()")
        return FakeStreamlit(f"{self._name}()", self._side)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _pipeline_has_run() -> bool:
    sys.path.insert(0, str(ROOT / "src"))
    from kbo_mlb import config
    return (config.INTERIM_DIR / "kbo_batting.csv").exists()


@unittest.skipUnless(_pipeline_has_run(),
                     "no data on disk; run the pipeline first")
class TestAppRenders(unittest.TestCase):
    def _run(self, side):
        saved = sys.modules.get("streamlit")
        sys.modules["streamlit"] = FakeStreamlit(side=side)
        try:
            code = compile(APP.read_text(), str(APP), "exec")
            exec(code, {"__name__": "__main__", "__file__": str(APP)})
        except SystemExit:
            self.fail(f"{side}: app called st.stop() on default settings")
        finally:
            if saved is None:
                sys.modules.pop("streamlit", None)
            else:
                sys.modules["streamlit"] = saved

    def test_hitter_page_renders(self):
        self._run("batting")

    def test_pitcher_page_renders(self):
        # The side that was broken. Pitcher projections are not trustworthy
        # and the app says so, but "not trustworthy" is a caveat, not a
        # traceback.
        self._run("pitching")


if __name__ == "__main__":
    unittest.main()
