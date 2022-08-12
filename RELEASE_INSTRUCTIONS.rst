===========================
M-LOOP Release Instructions
===========================

This document describes the process for generating new M-LOOP releases and is intended to be a reference for M-LOOP developers.

#. Create a PR to bump the version numbers in the required places.

   * In ``docs/conf.py``, update the following:

     * Update ``release = ...``.
     * If bumping the major or minor version number, update ``version = ...``.
     * Although it is commented out, it's probably wise to update ``html_title = ...``.

   * In ``mloop/__init__.py``, update the following:

     * Update ``__version__ = ...``.

   * In ``setup.py``, update the following:

     * Update ``download_url = 'https://github.com/michaelhush/M-LOOP/tarball/...``.

#. Merge the PR to bump the version numbers.
#. Create an environment to perform the release which includes an editable install of M-LOOP, as well as and the packages ``pip`` and ``twine``.

   * For example, this can be done with Anaconda Python using the following commands:

     * ``conda create --name m-loop-release-env --clone base``
     * ``conda activate m-loop-release-env``
     * ``conda install tensorflow``
     * ``conda install pip twine``
     * ``pip install -e .`` (Make sure to cd to M-LOOP project root directory first.)

#. Ensure you've checked out the master branch in your local repo and pulled the most recent changes, including the PR to bump the version numbers.
#. Stash any local changes to the repo, including any untracked files, with ``git stash --include-untracked`` to be sure that they won't accidentally be included in the release.
#. In the M-LOOP project root directory, run ``python setup.py sdist`` to make the source distribution.
#. Create a release on Github.

   * Go to `https://github.com/michaelhush/M-LOOP/releases <https://github.com/michaelhush/M-LOOP/releases>`_ and press "Draft a new release".
   * For the tag, put the version number as formatted in the last part of the ``download_url`` updated in ``setup.py``.

     * If the url's version number starts with a "v", be sure to include that here as well.

   * Set the title to "M-LOOP <version_number>" (without the quotes).
   * Click "Generate release notes" which will automatically fill out the description.
   * If necessary, edit the auto-generated description.
   * No need to upload any files - Github will automatically generate tar and zip files.
   * Double check that everything looks good.
   * Click "Publish release".

#. Do a dry run publishing to TestPyPI by running ``twine upload --repository testpypi dist/*`` in the M-LOOP project root directory.
#. If the previous step goes well, then run ``twine upload dist/*`` to publish to real PyPI.
#. If any local changes were stashed, you may want to reapply them with ``git stash apply``.
#. Build a new documentation version on readthedocs.

   * In particular a new version of "stable" should be built.
   * It may be wise to build a new version of "latest" as well since it may not have been built in a while.
