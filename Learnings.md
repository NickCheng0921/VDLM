### Pytest
Use pytest.ini to redirect main python path, prevents import errors
 - `testpaths` defines a specific test dir

pytest looks for `test_*.py`, or `*_test.py` and funcs inside with `test_` and treats each as a test