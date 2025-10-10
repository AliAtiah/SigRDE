def test_import_cli():
    import importlib
    mod = importlib.import_module('src.cli')
    assert hasattr(mod, 'main')


