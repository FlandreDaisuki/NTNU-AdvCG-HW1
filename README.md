# NTNU-AdvCG-HW1

detail: http://cg.csie.ntnu.edu.tw/AdvCG/Assignment1.pdf

According to [input file](http://cg.csie.ntnu.edu.tw/AdvCG/hw1_input.txt), it is a left-hand coordinate.

According to senpai, F(fov) represent for horizontal angle.

![](./depth2.png)

## Requirement
- pyenv
  - python 3.x
    - numpy
    - scipy

## Example result

![](./results/1489758169/result.png)

## Q&A

Q: ImportError: No module named '_tkinter'
```shell
# http://stackoverflow.com/questions/26357567/cannot-import-tkinter-after-installing-python-3-with-pyenv

$ sudo apt-get install tk-dev
$ pyenv install $YOUR_PYTHON3_VERSION
```