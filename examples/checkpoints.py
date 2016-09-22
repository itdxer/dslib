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
    logger.info("> Find outputs for {} steps".format(len(outputs)))
