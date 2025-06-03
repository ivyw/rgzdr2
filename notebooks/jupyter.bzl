load("@pypi//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

def jupyter_server(name, deps = []):
    py_binary(
        name = name,
        srcs = ["//notebooks:jupyter.py"],
        args = ["$(rlocationpaths @pypi//jupyterlab)"],
        main = "//notebooks:jupyter.py",
        deps = deps + [
            requirement("notebook"),
            "@pypi//jupyterlab",
            "@rules_python//python/runfiles",
            requirement("jupyterlab-code-formatter"),
            requirement("IPython"),
            requirement("ipympl"),
            requirement("ipywidgets"),
        ],
    )
