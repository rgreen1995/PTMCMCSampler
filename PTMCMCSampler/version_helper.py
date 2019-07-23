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
            return "%s <%s>" % (name.decode("utf-8"), email.decode("utf-8"))
        except Exception:
            return ""

    def get_build_date(self):
        """Return the current datetime
        """
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S +0000', time.gmtime())

    def get_last_commit_info():
        """Return the details of the last git commit
        """
        try:
            string = subprocess.check_output(
                ["git", "log", "-1", "--pretty=format:%H,%an,%ae])
            string = string.decode("utf-8").split(",")
            hash, username, email = string
            author = "%s <%s>" % (username, email)
            return hash, author
        except Exception:
            return ""

    def get_status():
        """Return the state of the git repository
        """
        git_diff = subprocess.check_output(
            ["git", "diff", "."]).decode("utf-8")

def get_git_status(git_path='git'):
    """Returns the state of the git working copy
    """
    status_output = subprocess.call((git_path, 'diff-files', '--quiet'))
    if status_output != 0:
        return 'UNCLEAN: Modified working tree'
    else:
        # check index for changes
        status_output = subprocess.call((git_path, 'diff-index', '--cached',
                                         '--quiet', 'HEAD'))
        if status_output != 0:
            return 'UNCLEAN: Modified index'
        else:
            return 'CLEAN: All modifications committed'

