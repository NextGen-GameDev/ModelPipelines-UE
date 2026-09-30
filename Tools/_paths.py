import os


def dev_dir():
    """Dev data folder: env MODELPIPELINES_DEV_DIR, else <repo>/../.pipelines-dev (absolute)."""
    env = os.environ.get("MODELPIPELINES_DEV_DIR")
    if env:
        return os.path.abspath(env)
    tools = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(tools)
    return os.path.abspath(os.path.join(repo, "..", ".pipelines-dev"))
