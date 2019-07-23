import os


def get_version_information():
    """Grab the version from the .version file
    """
    version_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), ".version")
    try:
        with open(version_file, "r") as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")

