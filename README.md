# NTNU-AdvCG-HW1

detail: http://cg.csie.ntnu.edu.tw/AdvCG/Assignment1.pdf

## Requirement
- pyenv
  - python 3.x
    - numpy
    - scipy

## Q&A

Q: ImportError: No module named '_tkinter'
```shell
# http://stackoverflow.com/questions/26357567/cannot-import-tkinter-after-installing-python-3-with-pyenv

$ sudo apt-get install tk-dev
$ pyenv install $YOUR_PYTHON3_VERSION
```