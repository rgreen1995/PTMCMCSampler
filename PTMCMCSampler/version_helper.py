import subprocess


class GitInformation(object):
    """Helper class to handle the git information
    """
    def __init__(self):
        self.last_commit_info = self.get_last_commit_info()
        self.hash = self.last_commit_info[0]
        self.author = self.last_commit_info[1]
        self.status = self.get_status()
        self.builder = self.get_build_name()
        self.build_date = self.get_build_date()

    def get_build_name(self):
        """Return the username and email of the current builder
        """
        try:
            name = subprocess.check_output(["git", "config", "user.name"])
            email = subprocess.check_output(["git", "config", "user.email"])
            name = name.strip()
            email = email.strip()
            return "%s <%s>" % (name.decode("utf-8"), email.decode("utf-8"))
        except Exception:
            return ""

    def get_build_date(self):
        """Return the current datetime
        """
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S +0000', time.gmtime())

    def get_last_commit_info(self):
        """Return the details of the last git commit
        """
        try:
            string = subprocess.check_output(
                ["git", "log", "-1", "--pretty=format:%H,%an,%ae"])
            string = string.decode("utf-8").split(",")
            hash, username, email = string
            author = "%s <%s>" % (username, email)
            return hash, author
        except Exception:
            return ""

    def get_status(self):
        """Return the state of the git repository
        """
        git_diff = subprocess.check_output(
            ["git", "diff", "."]).decode("utf-8")
        if git_diff:
            return "UNCLEAN: Modified working tree"
        return "CLEAN: All modifications committed"


class PackageInformation(object):
    """Helper class to handle package versions
    """
    def __init__(self):
        self.package_info = self.get_package_info()

    def get_package_info(self):
        """Return the package information
        """
        packages = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
        return packages
