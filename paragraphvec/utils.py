from os.path import join, dirname
import numpy as np
import time
import sys,logging

DATA_DIR = join(dirname(dirname(__file__)), 'data')
MODELS_DIR = join(dirname(dirname(__file__)), 'models')
MODEL_NAME = ("{:s}_model.{:s}.{:s}_contextsize.{:d}_numnoisewords.{:d}"
              "_vecdim.{:d}_batchsize.{:d}_lr.{:f}_epoch.{:d}_loss.{:f}"
              ".pth.tar")

class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

def init_logging(logfile_path):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S'   )
    fh = logging.FileHandler(logfile_path)
    # ch = logging.StreamHandler()
    ch = logging.StreamHandler(sys.stdout)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    logging.getLogger().addHandler(ch)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    return logging

def _print_progress(epoch_i, batch_i, num_batches):
    progress = round((batch_i + 1) / num_batches * 100)
    print("\rEpoch {:d}".format(epoch_i + 1), end='')
    sys.stdout.write(" - {:d}%".format(progress))
    sys.stdout.flush()

class Progbar(object):
    def __init__(self, target, width=30, batch_size = None, total_examples = None, verbose=1):
        '''
            @param target: total number of steps expected
        '''
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

        self.batch_size = batch_size
        self.last_batch = 0
        self.total_examples = total_examples
        self.start_time = time.time() - 0.00001
        self.last_time  = self.start_time

        self.logger = logging.getLogger()

    def update(self, current, values=[]):
        '''
        @param current: index of current step
        @param values: list of tuples (name, value_for_last_step).
        The progress bar will display averages for these values.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd' % (numdigits, numdigits, ) + ' (%.2f%%)['
            bar = barstr % (current, self.target, float(current)/float(self.target) * 100.0)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('.'*(prog_width-1))
                if current < self.target:
                    bar += '(-w-)'
                else:
                    bar += '(-v-)!!'
            bar += ('~' * (self.width-prog_width))
            bar += ']'
            # sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)

            # info = ''
            info = bar
            if current < self.target:
                info += ' - Run-time: %ds - ETA: %ds' % (now - self.start, eta)
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if k == 'perplexity' or k == 'PPL':
                    info += ' - %s: %.4f' % (k, np.exp(self.sum_values[k][0] / max(1, self.sum_values[k][1])))
                else:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))

            # update progress stats
            trained_word_count = self.batch_size * current  # only words in vocab & sampled
            new_trained_word_count = self.batch_size * (current - self.last_batch)  # only words in vocab & sampled
            current_time =time.time()
            elapsed = current_time - self.last_time
            info += " new processed %d words, %.0f words/s" % (new_trained_word_count, new_trained_word_count / elapsed)
            self.last_time = current_time
            self.last_batch = current


            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            # sys.stdout.write(info)
            # sys.stdout.flush()

            self.logger.info(info)

            if current >= self.target:
                sys.stdout.write("\n")


        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                # sys.stdout.write(info + "\n")
                self.logger.info(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)

    def clear(self):
        self.sum_values = {}
        self.unique_values = []
        self.total_width = 0
        self.seen_so_far = 0

