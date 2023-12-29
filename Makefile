install:
 #install
 pip install --upgrade pip && pip install -r requirements.txt
format:
 #format
 find . -name '*.py' -exec black {} +
lint:
 #lint
 pylint --disable=R,C ./*.py
test:
 #test
 python -m pytest -vv --cov=test