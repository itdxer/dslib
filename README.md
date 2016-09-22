# Useful tools for Data Scientist

## Installation

```bash
$ pip install dslib
```

## Logging

```python
>>> import time
>>> from dslib.logs import get_logger, logtime
>>>
>>> logger = get_logger()
>>> logger.info("Basic logging message")
[INFO    :22/09/2016 20:32:20] Basic logging message
>>>
>>> with logtime("Logging sleep function"):
...     time.sleep(5)
...
[INFO    :22/09/2016 20:32:22] [start:001] Start Logging sleep function
[INFO    :22/09/2016 20:32:27] [finish:001] Finish Logging sleep function (took 5.005 sec)
```

## Checkpoints

```python
from sklearn import datasets, linear_model, preprocessing
from dslib.logs import get_logger
from dslib.checkpoint import Checkpoint


logger = get_logger()


class ClassiyData(Checkpoint):
    def step_1(self, outputs):
        logger.info("Loading dataset")
        iris_dataset = datasets.load_iris()

        logger.info("Applying standard scaler")
        scaler = preprocessing.StandardScaler()
        data = scaler.fit_transform(iris_dataset.data)

        return scaler, data, iris_dataset.target

    def step_2(self, outputs):
        _, data, target = outputs['step_1']

        logger.info("Training model")
        logreg = linear_model.LogisticRegression()
        logreg.fit(data, target)

        return logreg


if __name__ == '__main__':
    logger.info("> Run classifier for the first time")
    classify_data = ClassiyData(
        name='classify-data',
        checkpoint_folder='.checkpoint',
        version=1
    )
    classify_data.run()

    logger.info("> Run classifier for the second time")
    classify_data.run(start_from=2)

    logger.info("> Load outputs")
    outputs = classify_data.load_outputs()
    logger.info("> Found outputs for {} steps".format(len(outputs)))
```

Output

```
[INFO    :22/09/2016 20:33:42] > Run classifier for the first time
[INFO    :22/09/2016 20:33:42] Checkpoint: #1
[INFO    :22/09/2016 20:33:42] Loading dataset
[INFO    :22/09/2016 20:33:42] Applying standard scaler
[INFO    :22/09/2016 20:33:42] Saving checkpoint into file: classify-data-v1-step1.pkl
[INFO    :22/09/2016 20:33:42] Checkpoint: #2
[INFO    :22/09/2016 20:33:42] Training model
[INFO    :22/09/2016 20:33:42] Saving checkpoint into file: classify-data-v1-step2.pkl
[INFO    :22/09/2016 20:33:42] > Run classifier for the second time
[INFO    :22/09/2016 20:33:42] Checkpoint: #1
[INFO    :22/09/2016 20:33:42] Loading checkpoint from file: classify-data-v1-step1.pkl
[INFO    :22/09/2016 20:33:42] Checkpoint: #2
[INFO    :22/09/2016 20:33:42] Training model
[INFO    :22/09/2016 20:33:42] Saving checkpoint into file: classify-data-v1-step2.pkl
[INFO    :22/09/2016 20:33:42] > Load outputs
[INFO    :22/09/2016 20:33:42] > Found outputs for 2 steps
```
