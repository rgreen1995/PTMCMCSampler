import os


def get_version_information():
    """Grab the version from the .version file
    """
    version_file = os.path.join(os.path.dirname(__file__), ".version")

    with open(version_file, "r") as f:
        f = f.readlines()
        f = [i.strip() for i in f]

    string = ""
    try:
        version = [i.split("= ")[1] for i in f if "last_release" in i][0]
        hash = [i.split("= ")[1] for i in f if "git_hash" in i][0]
        status = [i.split("= ")[1] for i in f if "git_status" in i][0]
        string += "%s: %s %s" % (version, status, hash)
    except Exception:
        pass
    return string
