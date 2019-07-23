import os
import sys
from setuptools import setup


version = "1.0.1"


def write_version_file(version):
    """Add the version number and the git hash to the file
    'PTMCMCSampler.__init__.py'

    Parameters
    ----------
    version: str
        the release version of the code that you are running
    """
    from PTMCMCSampler.version_helper import GitInformation, PackageInformation

    git_info = GitInformation()
    packages = PackageInformation()

    with open("PTMCMCSampler/.version", "w") as f:
        f.writelines(["# Generated automatically by PTMCMCSampler\n\n"])
        f.writelines(["last_release = %s\n" % (version)])
        f.writelines(["\ngit_hash = %s\n" % (git_info.hash)])
        f.writelines(["git_author = %s\n" % (git_info.author)])
        f.writelines(["git_status = %s\n" % (git_info.status)])
        f.writelines(["git_builder = %s\n" % (git_info.builder)])
        f.writelines(["git_build_date = %s\n" % (git_info.build_date)])
        f.writelines(['git_build_packages = """%s"""' % (packages.package_info)])
    return


write_version_file(version)


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()


setup(
    name="PTMCMCSampler",
    version=version,
    author="Justin A. Ellis",
    author_email="justin.ellis18@gmail.com",
    packages=["PTMCMCSampler", "PTMCMCSampler.proposals"],
    package_dir={"PTMCMCSampler": "PTMCMCSampler"},
    url="https://github.com/jellis18/PTMCMCSampler",
    license="MIT",
    zip_safe=False,
    description="Parallel tempering MCMC sampler written in Python",
    long_description=open("README.md").read()
    + "\n\n"
    + "---------\n\n"
    + open("HISTORY.md").read(),
    package_data={"": ["README.md", "HISTORY.md"]},
    install_requires=["numpy", "scipy"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
