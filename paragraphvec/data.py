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

        # initialize one generator with a shared state object
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

    def get_batch(self):
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

    def make_cum_table(self, power=0.75, domain=2**31 - 1):
        """
        Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.

        To draw a word index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if by bisect_left or ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.

        Called internally from 'build_vocab()'.
        """
        vocab_size = len(self._vocabulary)
        self.cum_table = np.multiarray.zeros(vocab_size, dtype=np.uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self._vocabulary[word].count**power for word in self._vocabulary]))
        cumulative = 0.0
        for word_index in range(vocab_size):
            cumulative += self._vocabulary[self.wv.index2word[word_index]].count**power
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

        # if model.negative:
        #     # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        #     word_indices = [predict_word.index]
        #     while len(word_indices) < model.negative + 1:
        #         # key part of speed up sampling
        #         w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
        #         if w != predict_word.index:
        #             word_indices.append(w)

    def _init_noise_distribution(self):
        # we use a unigram distribution raised to the 3/4rd power,
        # as proposed by T. Mikolov et al. in Distributed Representations
        # of Words and Phrases and their Compositionality
        probs = np.zeros(len(self._vocabulary) - 1)

        for word, freq in self._vocabulary.freqs.items():
            probs[self._word_to_index(word)] = freq

        probs = np.power(probs, 0.75)
        probs /= np.sum(probs)

        '''
        https://github.com/numpy/numpy/issues/7543
        numpy.random.choice() is very expensive:
            >>> %timeit -n 1 -r 1 [np.random.choice(50, 5, p=probs) for x in range(10000)]
                1 loop, best of 1: 331 ms per loop
            >>> %timeit -n 1 -r 1 [np.random.choice(50, 5, p=probs, replace=False) for x in range(10000)]
                1 loop, best of 1: 684 ms per loop
        numpy.random.multinomial() is cheaper:
            >>> %timeit -n 1 -r 1 [np.random.multinomial(1, probs).argmax() for x in range(10000)]
                1 loop, best of 1: 63.6 ms per loop
            >>> %timeit -n 1 -r 1 [np.random.multinomial(1, probs).argmax() for x in range(50000)]
                1 loop, best of 1: 279 ms per loop
        torch.utils.data.sampler is much better
            >>> %timeit -n 1 -r 1 [i for i in torch_sampler.WeightedRandomSampler(probs, 10000)]
                1 loop, best of 1: 5.3 ms per loop
            >>> %timeit -n 1 -r 1 [[i for i in torch_sampler.WeightedRandomSampler(probs, 5)]  for x in range(10000)]
                1 loop, best of 1: 135 ms per loop
        '''
        sampler_name = 'pytorch'
        if sampler_name == 'numpy':
            self._sample_noise = lambda: choice(
                probs.shape[0], self.num_noise_words, p=probs).tolist()
        elif sampler_name == 'pytorch':
            self._sample_noise = lambda: list(torch.utils.data.sampler.WeightedRandomSampler(
                probs, self.num_noise_words))

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

        # Get the starting point (index of doc and pos) to generate
        # Though here requires the state is synchronized, but only takes 1ms
        prev_doc_id, prev_in_doc_pos = self._state.update_state(
            self.dataset,
            self.batch_size,
            self.context_size,
            self._num_examples_in_doc)


        # takes around 1200~1500ms
        # generate the actual batch
        batch = _NCEBatch()

        # start_batch_time = current_milli_time()
        # start_document_time = current_milli_time()

        while len(batch) < self.batch_size:
            if prev_doc_id == len(self.dataset):
                # last document exhausted
                return self._batch_to_torch_data(batch)

            if prev_in_doc_pos <= (len(self.dataset[prev_doc_id].text) - 1
                                   - self.context_size):

                # start_time = current_milli_time()
                # more examples in the current document
                self._add_example_to_batch(prev_doc_id, prev_in_doc_pos, batch)
                prev_in_doc_pos += 1

                # current_time = current_milli_time()
                # print('\tGenerating one example time: %d ms, (%d, %d)' % (round(current_time - start_time), start_time, current_time))
            else:
                # go to the next document
                prev_doc_id += 1
                prev_in_doc_pos = self.context_size
                # print('\tGenerating one document time: %d ms, (%d, %d)' % (round(current_time - start_document_time), start_document_time, current_time))
                # start_document_time = current_milli_time()

        # current_time = current_milli_time()
        # print('\tGenerating one batch time: %d ms, (%d, %d)' % (round(current_time - start_batch_time), start_batch_time, current_time))

        torch_batch = self._batch_to_torch_data(batch)

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
        start_context_time = current_milli_time()

        doc = self.dataset[doc_id].text
        batch.doc_ids.append(doc_id)

        current_context = []
        for i in range(-self.context_size, self.context_size + 1):
            if i != 0:
                current_context.append(self._word_to_index(doc[in_doc_pos - i]))
        batch.context_ids.append(current_context)

        start_sample_time = current_milli_time()

        # Negative Sample: sample from the noise distribution, is the most time-consuming part, 99+% time of processing data
        current_noise = self._sample_noise()
        # the index 0 is target (central) word
        # current_noise.insert(0, self._word_to_index(doc[in_doc_pos]))
        current_noise = [self._word_to_index(doc[in_doc_pos])] + current_noise
        batch.target_noise_ids.append(current_noise)

        # current_time = current_milli_time()
        # example_time = round(current_time - start_context_time)
        # print('\tGenerating one example time: %d ms, scan=%.2f%%, sampling=%.2f%%)' % (example_time, float(start_sample_time-start_context_time)/example_time * 100.0, float(current_time-start_sample_time)/example_time * 100.0))

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
        # 'i' indicates a signed integer. These shared objects will be process and thread-safe.
        # initialize one new value for doc_id
        self._doc_id = multiprocessing.RawValue('i', 0)
        # initialize one new value for doc_pos (the first position is at pos=context_size)
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
