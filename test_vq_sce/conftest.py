""" https://stackoverflow.com/questions/55417214/
    phantom-tests-after-switching-from-unittest-testcase-to-tf-test-testcase """

def pytest_collection_modifyitems(session, config, items):
    items[:] = [item for item in items if item.name != 'test_session']