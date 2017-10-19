import multiprocessing
import os
import re
import signal
from math import ceil
from os.path import join

import numpy as np
import torch
from numpy.random import choice
from torchtext.data import Field, TabularDataset
import logging
from paragraphvec.utils import current_milli_time

from paragraphvec.utils import DATA_DIR

logger = logging.getLogger()

def load_dataset(file_name):
    """Loads contents from a file in the *data* directory into a
    torchtext.data.TabularDataset instance.
    """
    file_path = join(DATA_DIR, file_name)
    text_field = Field(pad_token=None, tokenize=_tokenize_str)

    dataset = TabularDataset(
        path=file_path,
        format='csv',
        fields=[('text', text_field)])

    text_field.build_vocab(dataset)
    return dataset


def _tokenize_str(str_):
    # keep only alphanumeric and punctations
    str_ = re.sub(r'[^A-Za-z0-9(),.!?\'`]', ' ', str_)
    # remove multiple whitespace characters
    str_ = re.sub(r'\s{2,}', ' ', str_)
    # punctations to tokens
    str_ = re.sub(r'\(', ' ( ', str_)
    str_ = re.sub(r'\)', ' ) ', str_)
    str_ = re.sub(r',', ' , ', str_)
    str_ = re.sub(r'\.', ' . ', str_)
    str_ = re.sub(r'!', ' ! ', str_)
    str_ = re.sub(r'\?', ' ? ', str_)
    # split contractions into multiple tokens
    str_ = re.sub(r'\'s', ' \'s', str_)
    str_ = re.sub(r'\'ve', ' \'ve', str_)
    str_ = re.sub(r'n\'t', ' n\'t', str_)
    str_ = re.sub(r'\'re', ' \'re', str_)
    str_ = re.sub(r'\'d', ' \'d', str_)
    str_ = re.sub(r'\'ll', ' \'ll', str_)
    # lower case
    return str_.strip().lower().split()


class NCEData(object):
    """An infinite, parallel (multiprocess) batch generator for
    noise-contrastive estimation of word vector models.

    Parameters
    ----------
    dataset: torchtext.data.TabularDataset
        Dataset from which examples are generated. A column labeled *text*
        is expected and should be comprised of a list of tokens. Each row
        should represent a single document.

    batch_size: int
        Number of examples per single gradient update.

    context_size: int
        Half the size of a neighbourhood of target words (i.e. how many
        words left and right are regarded as context).

    num_noise_words: int
        Number of noise words to sample from the noise distribution.

    max_size: int
        Maximum number of pre-generated batches.

    num_workers: int
        Number of jobs to run in parallel. If value is set to -1, total number
        of machine CPUs is used.
    """
    # code inspired by parallel generators in https://github.com/fchollet/keras
    def __init__(self, dataset, batch_size, context_size,
                 num_noise_words, max_size, num_workers):
        self.max_size = max_size

        self.num_workers = num_workers if num_workers != -1 else os.cpu_count()
        if self.num_workers is None:
            self.num_workers = 1

        self._generator = _NCEGenerator(
            dataset,
            batch_size,
            context_size,
            num_noise_words,
            _NCEGeneratorState(context_size))

        self.number_examples = self._generator.num_examples()
        self.number_documents = len(self._generator.dataset)


        logger.info('Actual num_workers = %d' % self.num_workers)
        logger.info('number_examples = %d' % self.number_examples)
        logger.info('number_documents = %d' % self.number_documents)


        self._queue = None
        self._stop_event = None
        self._processes = []

    def __len__(self):
        return len(self._generator)

    def vocabulary_size(self):
        return self._generator.vocabulary_size()

    def start(self):
        """Starts num_worker processes that generate batches of data."""
        self._queue = multiprocessing.Queue(maxsize=self.max_size)
        self._stop_event = multiprocessing.Event()

        for _ in range(self.num_workers):
            process = multiprocessing.Process(target=self._parallel_task)
            process.daemon = True
            self._processes.append(process)
            process.start()

    def _parallel_task(self):
        while not self._stop_event.is_set():
            try:
                # producer generates a new batch and pop it into queue
                batch = self._generator.next()
                # queue blocks a call to put() until a free slot is available
                self._queue.put(batch)
            except KeyboardInterrupt:
                self._stop_event.set()

    def get_generator(self):
        """Returns a generator that yields batches of data."""
        while self._is_running():
            yield self._queue.get()

    def stop(self):
        """Terminates all processes that were created with start()."""
        if self._is_running():
            self._stop_event.set()

        for process in self._processes:
            if process.is_alive():
                os.kill(process.pid, signal.SIGINT)
                process.join()

        if self._queue is not None:
            self._queue.close()

        self._queue = None
        self._stop_event = None
        self._processes = []

    def _is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()


class _NCEGenerator(object):
    """An infinite, process-safe batch generator for noise-contrastive
    estimation of word vector models.

    Parameters
    ----------
    state: paragraphvec.data._NCEGeneratorState
        Initial (indexing) state of the generator.

    For other parameters see the class NCEBatchPool.
    """
    def __init__(self, dataset, batch_size, context_size,
                 num_noise_words, state):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_noise_words = num_noise_words

        self._vocabulary = self.dataset.fields['text'].vocab
        self._sample_noise = None
        self._init_noise_distribution()
        self._state = state

    def _init_noise_distribution(self):
        # we use a unigram distribution raised to the 3/4rd power,
        # as proposed by T. Mikolov et al. in Distributed Representations
        # of Words and Phrases and their Compositionality
        probs = np.zeros(len(self._vocabulary) - 1)

        for word, freq in self._vocabulary.freqs.items():
            probs[self._word_to_index(word)] = freq

        probs = np.power(probs, 0.75)
        probs /= np.sum(probs)

        self._sample_noise = lambda: choice(
            probs.shape[0], self.num_noise_words, p=probs).tolist()

    def __len__(self):
        num_examples = sum(self._num_examples_in_doc(d) for d in self.dataset)
        return ceil(num_examples / self.batch_size)

    def num_examples(self):
        return sum(self._num_examples_in_doc(d) for d in self.dataset)

    def vocabulary_size(self):
        return len(self._vocabulary) - 1

    def next(self):
        """Updates state for the next process in a process-safe manner
        and generates the current batch."""
        prev_doc_id, prev_in_doc_pos = self._state.update_state(
            self.dataset,
            self.batch_size,
            self.context_size,
            self._num_examples_in_doc)

        # generate the actual batch
        batch = _NCEBatch()

        start_time = current_milli_time()

        while len(batch) < self.batch_size:
            if prev_doc_id == len(self.dataset):
                # last document exhausted
                return self._batch_to_torch_data(batch)
            if prev_in_doc_pos <= (len(self.dataset[prev_doc_id].text) - 1
                                   - self.context_size):
                # more examples in the current document
                self._add_example_to_batch(prev_doc_id, prev_in_doc_pos, batch)
                prev_in_doc_pos += 1
            else:
                # go to the next document
                prev_doc_id += 1
                prev_in_doc_pos = self.context_size

        current_time = current_milli_time()
        print('generating batch time: %d ms, (%d, %d)' % (round(current_time - start_time), start_time, current_time))

        start_time = current_milli_time()
        torch_batch = self._batch_to_torch_data(batch)
        current_time = current_milli_time()
        print('transfer batch to Torch: %d ms, (%d, %d)' % (round(current_time - start_time), start_time, current_time))

        return torch_batch

    def _num_examples_in_doc(self, doc, in_doc_pos=None):
        if in_doc_pos is not None:
            # number of remaining
            if len(doc.text) - in_doc_pos >= self.context_size + 1:
                return len(doc.text) - in_doc_pos - self.context_size
            return 0

        if len(doc.text) >= 2 * self.context_size + 1:
            # total number
            return len(doc.text) - 2 * self.context_size
        return 0

    def _add_example_to_batch(self, doc_id, in_doc_pos, batch):
        doc = self.dataset[doc_id].text
        batch.doc_ids.append(doc_id)

        current_context = []
        for i in range(-self.context_size, self.context_size + 1):
            if i != 0:
                current_context.append(self._word_to_index(doc[in_doc_pos - i]))
        batch.context_ids.append(current_context)

        # sample from the noise distribution
        current_noise = self._sample_noise()
        # the index 0 is target (central) word
        current_noise.insert(0, self._word_to_index(doc[in_doc_pos]))
        batch.target_noise_ids.append(current_noise)

    def _word_to_index(self, word):
        return self._vocabulary.stoi[word] - 1

    @staticmethod
    def _batch_to_torch_data(batch):
        batch.context_ids = torch.LongTensor(batch.context_ids)
        batch.doc_ids = torch.LongTensor(batch.doc_ids)
        batch.target_noise_ids = torch.LongTensor(batch.target_noise_ids)

        if torch.cuda.is_available():
            batch.context_ids = batch.context_ids.cuda()
            batch.doc_ids = batch.doc_ids.cuda()
            batch.target_noise_ids = batch.target_noise_ids.cuda()

        return batch


class _NCEGeneratorState(object):
    """Batch generator state that is represented with a document id and
    in-document position. It abstracts a process-safe indexing mechanism."""
    def __init__(self, context_size):
        # use raw values because both indices have
        # to manually be locked together
        self._doc_id = multiprocessing.RawValue('i', 0)
        self._in_doc_pos = multiprocessing.RawValue('i', context_size)
        self._lock = multiprocessing.Lock()

    def update_state(self, dataset, batch_size,
                     context_size, num_examples_in_doc):
        """Returns current indices and computes new indices for the
        next process."""
        with self._lock:
            doc_id = self._doc_id.value
            in_doc_pos = self._in_doc_pos.value
            self._advance_indices(
                dataset, batch_size, context_size, num_examples_in_doc)
            return doc_id, in_doc_pos

    def _advance_indices(self, dataset, batch_size,
                         context_size, num_examples_in_doc):
        num_examples = num_examples_in_doc(
            dataset[self._doc_id.value], self._in_doc_pos.value)

        if num_examples > batch_size:
            # more examples in the current document
            self._in_doc_pos.value += batch_size
            return

        if num_examples == batch_size:
            # just enough examples in the current document
            if self._doc_id.value < len(dataset) - 1:
                self._doc_id.value += 1
            else:
                self._doc_id.value = 0
            self._in_doc_pos.value = context_size
            return

        while num_examples < batch_size:
            if self._doc_id.value == len(dataset) - 1:
                # last document: reset indices
                self._doc_id.value = 0
                self._in_doc_pos.value = context_size
                return

            self._doc_id.value += 1
            num_examples += num_examples_in_doc(
                dataset[self._doc_id.value])

        self._in_doc_pos.value = (len(dataset[self._doc_id.value].text)
                                  - context_size
                                  - (num_examples - batch_size))


class _NCEBatch(object):
    def __init__(self):
        self.context_ids = []
        self.doc_ids = []
        self.target_noise_ids = []

    def __len__(self):
        return len(self.doc_ids)
