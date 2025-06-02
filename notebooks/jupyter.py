"""Get Jupyter notebooks working in Bazel.

Source: https://github.com/bazel-contrib/rules_python/issues/63#issuecomment-2505695042
"""

import sys
import os

from python.runfiles import Runfiles

# Find the real location of the jupyterlab __init__.py file. The first argument is
# the exe name, all the other arguments are the files in the jupyterlab package.
# Unfortunately we can't ask bazel for the directory, only the location of all the
# files. We only need one though and will ignore the rest.
jupyterlab_init_file = next(
    x for x in sys.argv if x.endswith("site-packages/jupyterlab/__init__.py")
)

r = Runfiles.Create()
real_location = r.Rlocation(jupyterlab_init_file)

# Set the JUPYTERLAB_DIR because bazel will not set this for us. Without this, the
# server won't be able to find any extensions and will fail.
os.environ["JUPYTERLAB_DIR"] = os.path.dirname(real_location)

# Set the working directory to the bazel workspace root so we start there, instead
# of the runfiles directory.
os.chdir(os.environ["BUILD_WORKSPACE_DIRECTORY"])

# Remove all the jupyterlab file paths because the jupyter server will try to open
# them as notebook files, but they aren't real paths so it will fail.
sys.argv = [sys.argv[0]]

# We have to import the main function here, otherwise our environment manipulation
# above will not apply to it.
from notebook.app import main

sys.exit(main())
