import os
import pickle
from collections import namedtuple

from .logs import get_logger, yellow


def chunker(sequence, size):
    size = int(size)
    for position in range(0, len(sequence), size):
        yield sequence[position:position + size]


def save_pickle(object_, filepath):
    with open(filepath, 'wb') as f:
        pickled_object = pickle.dumps(object_)

        for object_chunk in chunker(pickled_object, size=1e10):
            f.write(object_chunk)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


class Checkpoint(object):
    def __init__(self, name, checkpoint_folder, version=1, logger=None):
        if logger is None:
            logger = get_logger()

        self.name = name
        self.checkpoint_folder = checkpoint_folder
        self.version = version
        self.logger = logger

        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)

    def iter_steps(self):
        checkpoint_folder = self.checkpoint_folder
        version = self.version
        name = self.name

        Step = namedtuple("Step", "step_id method_name filename filepath")

        methods = dir(self)
        step_methods = [m for m in methods if m.startswith('step_')]
        step_ids = [int(m.split('_')[1]) for m in step_methods]
        step_ids.sort()

        for step_method, step_id in zip(step_methods, step_ids):
            checkpoint_filename = "{name}-v{version}-step{step_id}.pkl".format(
                name=name, version=version, step_id=step_id
            )
            checkpoint_fullpath = os.path.join(checkpoint_folder,
                                               checkpoint_filename)
            yield Step(
                step_id=step_id,
                method_name=step_method,
                filename=checkpoint_filename,
                filepath=checkpoint_fullpath,
            )

    def run(self, start_from=0):
        outputs = {}
        for step in self.iter_steps():
            self.logger.info(yellow("Checkpoint: #{}".format(step.step_id)))

            if step.step_id < start_from and os.path.exists(step.filepath):
                self.logger.info("Loading checkpoint from file: {}"
                                 "".format(step.filename))
                outputs[step.method_name] = load_pickle(step.filepath)

            else:
                method = getattr(self, step.method_name)
                output = method(outputs)
                outputs[step.method_name] = output

                self.logger.info("Saving checkpoint into file: {}"
                                 "".format(step.filename))

                save_pickle(output, step.filepath)

    def load_outputs(self):
        outputs = {}
        for step in self.iter_steps():
            if os.path.exists(step.filepath):
                outputs[step.method_name] = load_pickle(step.filepath)
        return outputs
