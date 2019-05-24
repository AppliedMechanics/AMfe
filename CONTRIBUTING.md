CONTRIBUTING
============

This document gives an overview over how you can contribute to the AMfe.


Issue Reporting
---------------

The easiest thing you can contribute is reporting issues.
If you realize a bug you can report the issue on the issues section.
Please write an exact description of a bug.
Include a minimal example or better a minimal unit test in the issue description
which would fail due to the bug. This would be the best practise to report
an issue.  

Bug Fixing
----------

You are encouraged to fix the bug by yourself. Indicate your plan by assigning
the issue to you. Try to fix the bug locally. When you are sure that you can fix
the bug, create a merge request from master. Add your code in the merge request.
Please follow the guidelines below for good coding style and not forgetting things.
Add a comment in the merge request that you have finished your code and that it is
ready for review. A reviewer will review your code and add comments if you still 
need to enhance your code for being accepted for merge.
Note: No merge request will be accepted without a unit test and docstrings!

Checklist:

- [ ] Write a unit test in tests folder for your bugfix
- [ ] Check if your unit test has good code coverage!
- [ ] Check if your code is PEP8 conform. Most IDEs have settings that can help you checking this.
- [ ] Check if your new code has a docstring following the numpydoc conventions
- [ ] Check if your function's docstring is added in the docs (reference API docs)
- [ ] If it is a new function, tag with .. versionadded::X.Y.Z with X.Y.Z the version number of the next release 
which can be found in setup.py
- [ ] Add your name with a comment what you have fixed in the THANKS.txt (This is voluntary,
but really desired to give every author credit)
- [ ] Add a release-note in the release-notes.txt (in docs/release)
- [ ] If it is a new major function, perhaps add a tutorial/example etc. in the main documentation
- [ ] Check that the code you use is BSD license compatible! If it is your own code, note that
you accept the BSD license conditions
- [ ] If you used compiled code or a package from a private repo, is it correctly introduced in setup.py?

Afterwards you can ask a reviewer to review your code. Afterwards, the maintainer (currently Christian Meyer)
decides over the final merge. 

New Features
------------

In general, AMfe is a code that is often used for research. It also provides methods that are quite new or
still in research. However, if you try out a new method you use, this is ok in your own private branches,
but it is not intended to become directly a main feature of AMfe. AMfe does only contain features that
are tested and useful in that sense that research results show its benefit. Thus the usual workflow is:
Checkout your private branche, do your research, make your publications and if you can show, it is a useful
method, you can propose a separate merge request where only the new well working method is proposed to merge.
This is highly desired, of course. In contrast, untested or even not working features are not welcome.

If you plan to add a new larger feature to AMfe, we will make a new feature branch with the developer
branch as parent. There, many Merge Requests can be done until the feature is working.


Review
------

Each Merge Request into developer branch or master merge MUST be reviewed. Other branches are review free.
Reviewers should ask themselves the following questions when reviewing:

- Was this change discussed in the community in case of new functions/features.
- Is the feature well known in literature or is it new research? In the latter case, are there publications or
any tests that show the performance of the code
- Is the code decoupled from other code (e.g. parameters are low-level datatypes (arrays, floats etc.))?
- Is the encapsulation of the classes ensured (private properties, public methods)?
- Does the code meet the checklist above?
- Is there any code that is duplicated at another part? In this case, can it be refactored?
- Are the new functions/features tested properly by a unittest?
- Are there docstrings for the new fuctions/features?


Testing
-------

To test your code use nosetests. You need to install the nose python package.
Afterwards you can run nosetests inside the AMfe directory.
Write a test in your test_folder, check if nosetests recognizes it (you have to ensure that
your methods start with the 'test_' prefix) and if it passes.

Use coverage.py to check if your tests cover all lines of your new developed/bugfixed functions.
