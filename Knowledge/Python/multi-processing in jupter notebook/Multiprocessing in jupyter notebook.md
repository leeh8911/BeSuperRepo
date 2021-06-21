# Jupyter notebook에서 multiprocessing 하기!

주피터 노트북(Jupyter notebook)상에서 멀티 프로세싱(multiprocessing)을 활용하기 위해서는 실제 작업할 함수를 외부 모듈(즉, `*.py`)로 만들어야 한다.

아래 예제의 경우 1.1은 노트북 상에 함수를 선언하고, 해당 함수를 통해 pool에 넣어주면 더 이상 실행되지 않고 멈춰있게 된다.

하지만 1.2의 경우처럼 사용하고자 하는 함수를 외부에 만들어 놓고 불러오면(import하면) 제대로 작동하게 된다.

![그림](https://raw.githubusercontent.com/leeh8911/BeSuperRepo/main/Knowledge/Python/multi-processing%20in%20jupter%20notebook/file_tree.PNG?token=AM55F3JPB3EFWAYKGYV5CF3A2AYPA)

## 잘 안되는 경우


```python
from multiprocessing import Pool
import multiprocessing as multi
from tqdm import tqdm
```


```python
def test_function(src):
    return src**2
```


```python
with Pool(4) as pool:
    imap = pool.imap(test_function, list(range(10)))
    results = list(tqdm(imap, total=10, desc="processing"))
print(results)
```

    output
    processing:   0%|                                                                               | 0/10 [00:00<?, ?it/s]

## 잘 되는 경우


```python
from multiprocessing import Pool
import multiprocessing as multi
from tqdm import tqdm
```


```python
from utils.test import test_function
```


```python
with Pool(4) as pool:
    imap = pool.imap(test_function, list(range(10)))
    results = list(tqdm(imap, total=10, desc="processing"))
print(results)
```

    output
    processing: 100%|█████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 147.06it/s]
    
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]


​    
​    

# Reference
1. [python multiprocessing 을 Windows jupyter 에서 실행시키기!](https://devkyu.tistory.com/m/920)
