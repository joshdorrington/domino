version=$(cat setup.py | grep version= | cut -d \' -f2)
python -m build
python -m twine upload --repository pypi dist/*$version* --verbose
pip install domino-composite==$version
pip install domino-composite==$version

git add ./
git commit -m "$1"
git push origin master
