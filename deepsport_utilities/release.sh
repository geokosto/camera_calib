#!/bin/bash

PACKAGE_NAME="deepsport_utilities"

# build documentation
pdoc $PACKAGE_NAME -o docs/ -c latex_math=True --force --html

# Extract version number from setup.py
RELEASE_NUM=`grep "version=" setup.py | cut -d\" -f2 | cut -d\' -f2`
echo "RELEASE_NUM=$RELEASE_NUM"

# Push to PyPi
python setup.py sdist
echo "uploading 'dist/${PACKAGE_NAME}-${RELEASE_NUM}.tar.gz'"
twine upload dist/${PACKAGE_NAME}-${RELEASE_NUM}.tar.gz

# Tag in Git and push to remote
git tag -f $RELEASE_NUM -m "Tagging release $RELEASE_NUM"
git push -f --tags

