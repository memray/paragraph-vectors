import os
import time
from os import remove
from os.path import join
from sys import stdout, float_info

import fire
import torch
from torch.optim import Adam

from paragraphvec.data import load_dataset, NCEData
from paragraphvec.loss import NegativeSampling
from paragraphvec.models import DistributedMemory
from paragraphvec.utils import *
from paragraphvec.utils import _print_progress


def start(data_file_name,
          context_size,
          num_noise_words,
          vec_dim,
          num_epochs,
          batch_size,
          lr,
          model_ver='dm',
          vec_combine_method='sum',
          save_all=False,
          max_generated_batches=5,
          num_workers=1):
    """Trains a new model. The latest checkpoint and the best performing
    model are saved in the *models* directory.

    Parameters
    ----------
    data_file_name: str
        Name of a file in the *data* directory.

    context_size: int
        Half the size of a neighbourhood of target words (i.e. how many
        words left and right are regarded as context).

    num_noise_words: int
        Number of noise words to sample from the noise distribution.

    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_epochs: int
        Number of iterations to train the model (i.e. number
        of times every example is seen during training).

    batch_size: int
        Number of examples per single gradient update.

    lr: float
        Learning rate of the Adam optimizer.

    model_ver: str, one of ('dm', 'dbow'), default='dm'
        Version of the model as proposed by Q. V. Le et al., Distributed
        Representations of Sentences and Documents. 'dm' stands for
        Distributed Memory, 'dbow' stands for Distributed Bag Of Words.
        Currently only the 'dm' version is implemented.

        But according to [doc2vec paper](http://proceedings.mlr.press/v32/le14.pdf) and [empirical analysis](https://arxiv.org/pdf/1607.05368.pdf), 'dbow' is running better

    vec_combine_method: str, one of ('sum', 'concat'), default='sum'
        Method for combining paragraph and word vectors in the 'dm' model.
        Currently only the 'sum' operation is implemented.

    save_all: bool, default=False
        Indicates whether a checkpoint is saved after each epoch.
        If false, only the best performing model is saved.

    max_generated_batches: int, default=5
        Maximum number of pre-generated batches.

    num_workers: int, default=1
        Number of batch generator jobs to run in parallel. If value is set
        to -1 number of machine cores are used.
    """
    assert model_ver in ('dm', 'dbow')
    assert vec_combine_method in ('sum', 'concat')

    init_logging('../experiments/experiments.{0}.id={1}.log'.format('doc2vec', time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))))

    dataset = load_dataset(data_file_name)
    nce_data = NCEData(
        dataset,
        batch_size,
        context_size,
        num_noise_words,
        max_generated_batches,
        num_workers)
    nce_data.start()

    try:
        _run(data_file_name, dataset, nce_data.get_generator(), len(nce_data),
             nce_data.vocabulary_size(), nce_data.number_examples, context_size, num_noise_words, vec_dim,
             num_epochs, batch_size, lr, model_ver, vec_combine_method,
             save_all)
    except KeyboardInterrupt:
        nce_data.stop()

def _run(data_file_name,
         dataset,
         data_generator,
         num_batches,
         vocabulary_size,
         number_examples,
         context_size,
         num_noise_words,
         vec_dim,
         num_epochs,
         batch_size,
         lr,
         model_ver,
         vec_combine_method,
         save_all):
    '''
    Averagely, the time consumption:
    max_generated_batches = 5
        CPU:
            backward time: 600~650 ms
            sampling time: 1 ms
            forward time:  5~7 ms
        GPU:
            backward time: 3 ms
            sampling time: 72 ms
            forward time:  1~2 ms
    Should rewrite sampling to speed up on GPU

    DocTag2Vec on CPU:
        121882 words/s, 8 workers
        processing one document time = 650~850 ms
        training on 173403030 raw words (68590824 effective words) took 646.2s, 106138 effective words/s
    '''

    model = DistributedMemory(
        vec_dim,
        num_docs=len(dataset),
        num_words=vocabulary_size)

    cost_func = NegativeSampling()
    optimizer = Adam(params=model.parameters(), lr=lr)
    logger = logging.getLogger('root')

    if torch.cuda.is_available():
        model.cuda()
        logger.info("Running on GPU - CUDA")
    else:
        logger.info("Running on CPU")

    logger.info("Dataset comprised of {:d} documents.".format(len(dataset)))
    logger.info("Vocabulary size is {:d}.\n".format(vocabulary_size))
    logger.info("Training started.")

    best_loss = float_info.max
    prev_model_file_path = ""

    current_milli_time = lambda: int(round(time.time() * 1000))

    progbar = Progbar(num_batches, batch_size=batch_size, total_examples = number_examples)

    for epoch_i in range(num_epochs):
        epoch_start_time = current_milli_time
        loss = []

        for batch_i in range(num_batches):
            start_time = current_milli_time()
            batch = next(data_generator)
            print('data-prepare time: %d ms' % round(current_milli_time() - start_time))

            start_time = current_milli_time()
            x = model.forward(
                batch.context_ids,
                batch.doc_ids,
                batch.target_noise_ids)
            x = cost_func.forward(x)
            loss.append(x.data[0])
            print('forward time: %d ms' % round(current_milli_time() - start_time))

            start_time = current_milli_time()
            model.zero_grad()
            x.backward()
            optimizer.step()
            print('backward time: %d ms' % round(current_milli_time() - start_time))

            progbar.update(batch_i, )
            # _print_progress(epoch_i, batch_i, num_batches)

        # end of epoch
        loss = torch.mean(torch.FloatTensor(loss))
        is_best_loss = loss < best_loss
        best_loss = min(loss, best_loss)
        progbar.update(batch_i, [('loss', loss), ('best_loss', best_loss)])

        model_file_name = MODEL_NAME.format(
            data_file_name[:-4],
            model_ver,
            vec_combine_method,
            context_size,
            num_noise_words,
            vec_dim,
            batch_size,
            lr,
            epoch_i + 1,
            loss)
        model_file_path = join(MODELS_DIR, model_file_name)
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        state = {
            'epoch': epoch_i + 1,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optimizer.state_dict()
        }
        if save_all:
            torch.save(state, model_file_path)
        elif is_best_loss:
            try:
                remove(prev_model_file_path)
            except FileNotFoundError:
                pass
            torch.save(state, model_file_path)
            prev_model_file_path = model_file_path

        epoch_total_time = round(time.time() - epoch_start_time)
        logger.info(" ({:d}s) - loss: {:.4f}".format(epoch_total_time, loss))


if __name__ == '__main__':
    args = "--data_file_name 'doc2vec-pytorch_mag_fos=ir.csv' --num_epochs 10 --batch_size 512 --context_size 5 --num_noise_words 5 --vec_dim 300 --lr 1e-4 --save_all true --max_generated_batches 5120 --num_workers -1".split()
    fire.Fire(start, args)
