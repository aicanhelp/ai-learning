"""
Config controlling hyperparameters for pre-training ELECTRA.
Refactored from the implementation of Google
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from electrax.util import training_utils
import tensorflow.compat.v1 as tf

from dataclasses import dataclass
from datetime import datetime
from aiharness import harnessutils as utils
from electrax.util import tfutils
from aiharness.configuration import ArgType, Arguments
from aiharness import harnessutils as utils
import os

CLS = '[CLS]'
SEP = '[SEP]'
PAD = '[PAD]'
data_log = utils.getLogger('data')
pretrain_log = utils.getLogger('pretrain')
finetune_log = utils.getLogger('finetune')


@dataclass
class DataGeneratorConfig:
    corpus_dir: ArgType = ArgType('dataset', 'Location of pre-training text files.')
    vocab_file: ArgType = ArgType('dataset/vocab.txt', 'Location of vocabulary file.')
    output_dir: ArgType = ArgType('build/dataset/tf', 'Where to write out the tfrecords.')
    max_seq_length: ArgType = ArgType(128, 'Number of tokens per example.')
    num_process: ArgType = ArgType(4, 'Parallelize across multiple processes.')
    blanks_separate_docs: ArgType = ArgType(True, 'Whether blank lines indicate document boundaries.')
    do_lower_case: ArgType = ArgType(True, 'Lower case input text.')
    no_lower_case: ArgType = ArgType(False, "Don't lower case input text.")
    flatten: ArgType = ArgType(False, "whether the corpus file is zip/tar file.")

    def update(self):
        tfutils.rmkdir(self.output_dir)


def get_data_config():
    return Arguments(DataGeneratorConfig()).parse().update()


class PretrainingConfig(object):
    """Defines pre-training hyperparameters."""

    def __init__(self, model_name, data_dir, **kwargs):
        self.model_name = model_name
        self.debug = False  # debug mode for quickly running things
        self.do_train = True  # pre-train ELECTRA
        self.do_eval = False  # evaluate generator/discriminator on unlabeled data

        # loss functions
        self.electra_objective = True  # if False, use the BERT objective instead
        self.gen_weight = 1.0  # masked language modeling / generator loss
        self.disc_weight = 50.0  # discriminator loss
        self.mask_prob = 0.15  # percent of input tokens to mask out / replace

        # optimization
        self.learning_rate = 5e-4
        self.lr_decay_power = 1.0  # linear weight decay by default
        self.weight_decay_rate = 0.01
        self.num_warmup_steps = 10000

        # training settings
        self.iterations_per_loop = 200
        self.save_checkpoints_steps = 1000
        self.num_train_steps = 1000000
        self.num_eval_steps = 100

        # model settings
        self.model_size = "small"  # one of "small", "base", or "large"
        # override the default transformer hparams for the provided model size; see
        # modeling.BertConfig for the possible hparams and util.training_utils for
        # the defaults
        self.model_hparam_overrides = (
            kwargs["model_hparam_overrides"]
            if "model_hparam_overrides" in kwargs else {})
        self.embedding_size = None  # bert hidden size by default
        self.vocab_size = 30522  # number of tokens in the vocabulary
        self.do_lower_case = True  # lowercase the input?

        # generator settings
        self.uniform_generator = False  # generator is uniform at random
        self.untied_generator_embeddings = False  # tie generator/discriminator
        # token embeddings?
        self.untied_generator = True  # tie all generator/discriminator weights?
        self.generator_layers = 1.0  # frac of discriminator layers for generator
        self.generator_hidden_size = 0.25  # frac of discrim hidden size for gen
        self.disallow_correct = False  # force the generator to sample incorrect
        # tokens (so 15% of tokens are always
        # fake)
        self.temperature = 1.0  # temperature for sampling from generator

        # batch sizes
        self.max_seq_length = 128
        self.train_batch_size = 128
        self.eval_batch_size = 128

        # TPU settings
        self.use_tpu = False
        self.num_tpu_cores = 1
        self.tpu_job_name = None
        self.tpu_name = None  # cloud TPU to use for training
        self.tpu_zone = None  # GCE zone where the Cloud TPU is located in
        self.gcp_project = None  # project name for the Cloud TPU-enabled project

        # default locations of data files
        self.pretrain_tfrecords = os.path.join(
            data_dir, "pretrain_tfrecords/pretrain_data.tfrecord*")
        self.vocab_file = os.path.join(data_dir, "vocab.txt")
        self.model_dir = os.path.join(data_dir, "models", model_name)
        results_dir = os.path.join(self.model_dir, "results")
        self.results_txt = os.path.join(results_dir, "unsup_results.txt")
        self.results_pkl = os.path.join(results_dir, "unsup_results.pkl")

        # update defaults with passed-in hyperparameters
        self.update(kwargs)

        self.max_predictions_per_seq = int((self.mask_prob + 0.005) *
                                           self.max_seq_length)

        # debug-mode settings
        if self.debug:
            self.train_batch_size = 8
            self.num_train_steps = 20
            self.eval_batch_size = 4
            self.iterations_per_loop = 1
            self.num_eval_steps = 2

        # defaults for different-sized model
        if self.model_size == "small":
            self.embedding_size = 128
        # Here are the hyperparameters we used for larger models; see Table 6 in the
        # paper for the full hyperparameters
        # else:
        #   self.max_seq_length = 512
        #   self.learning_rate = 2e-4
        #   if self.model_size == "base":
        #     self.embedding_size = 768
        #     self.generator_hidden_size = 0.33333
        #     self.train_batch_size = 256
        #   else:
        #     self.embedding_size = 1024
        #     self.mask_prob = 0.25
        #     self.train_batch_size = 2048

        # passed-in-arguments override (for example) debug-mode defaults
        self.update(kwargs)
        self.bert_config = None

    def build(self):
        self.bert_config = training_utils.get_bert_config(self)
        if self.debug:
            self.bert_config.num_hidden_layers = 3
            self.bert_config.hidden_size = 144
            self.bert_config.intermediate_size = 144 * 4
            self.bert_config.num_attention_heads = 4

        self.embedding_size = (
            self.bert_config.hidden_size if self.embedding_size is None else
            self.embedding_size)
        return self

    def update(self, kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                raise ValueError("Unknown hparam " + k)
            self.__dict__[k] = v


class FinetuningConfig(object):
    """Fine-tuning hyperparameters."""

    def __init__(self, model_name, data_dir, **kwargs):
        # general
        self.model_name = model_name
        self.debug = False  # debug mode for quickly running things
        self.log_examples = False  # print out some train examples for debugging
        self.num_trials = 1  # how many train+eval runs to perform
        self.do_train = True  # train a model
        self.do_eval = True  # evaluate the model
        self.do_test = True  # evaluate on the test set
        self.keep_all_models = True  # if False, only keep the last trial's ckpt

        # model
        self.model_size = "small"  # one of "small", "base", or "large"
        self.task_names = ["chunk"]  # which tasks to learn
        # override the default transformer hparams for the provided model size; see
        # modeling.BertConfig for the possible hparams and util.training_utils for
        # the defaults
        self.model_hparam_overrides = (
            kwargs["model_hparam_overrides"]
            if "model_hparam_overrides" in kwargs else {})
        self.embedding_size = None  # bert hidden size by default
        self.vocab_size = 30522  # number of tokens in the vocabulary
        self.do_lower_case = True

        # training
        self.learning_rate = 1e-4
        self.weight_decay_rate = 0.01
        self.layerwise_lr_decay = 0.8  # if > 0, the learning rate for a layer is
        # lr * lr_decay^(depth - max_depth) i.e.,
        # shallower layers have lower learning rates
        self.num_train_epochs = 3.0  # passes over the dataset during training
        self.warmup_proportion = 0.1  # how much of training to warm up the LR for
        self.save_checkpoints_steps = 1000000
        self.iterations_per_loop = 1000
        self.use_tfrecords_if_existing = True  # don't make tfrecords and write them
        # to disc if existing ones are found

        # writing model outputs to disc
        self.write_test_outputs = False  # whether to write test set outputs,
        # currently supported for GLUE + SQuAD 2.0
        self.n_writes_test = 5  # write test set predictions for the first n trials

        # sizing
        self.max_seq_length = 128
        self.train_batch_size = 32
        self.eval_batch_size = 32
        self.predict_batch_size = 32
        self.double_unordered = True  # for tasks like paraphrase where sentence
        # order doesn't matter, train the model on
        # on both sentence orderings for each example
        # for qa tasks
        self.max_query_length = 64  # max tokens in q as opposed to context
        self.doc_stride = 128  # stride when splitting doc into multiple examples
        self.n_best_size = 20  # number of predictions per example to save
        self.max_answer_length = 30  # filter out answers longer than this length
        self.answerable_classifier = True  # answerable classifier for SQuAD 2.0
        self.answerable_uses_start_logits = True  # more advanced answerable
        # classifier using predicted start
        self.answerable_weight = 0.5  # weight for answerability loss
        self.joint_prediction = True  # jointly predict the start and end positions
        # of the answer span
        self.beam_size = 20  # beam size when doing joint predictions
        self.qa_na_threshold = -2.75  # threshold for "no answer" when writing SQuAD
        # 2.0 test outputs

        # TPU settings
        self.use_tpu = False
        self.num_tpu_cores = 1
        self.tpu_job_name = None
        self.tpu_name = None  # cloud TPU to use for training
        self.tpu_zone = None  # GCE zone where the Cloud TPU is located in
        self.gcp_project = None  # project name for the Cloud TPU-enabled project

        # default locations of data files
        self.data_dir = data_dir
        pretrained_model_dir = os.path.join(data_dir, "models", model_name)
        self.raw_data_dir = os.path.join(data_dir, "finetuning_data", "{:}").format
        self.vocab_file = os.path.join(pretrained_model_dir, "vocab.txt")
        if not tf.io.gfile.exists(self.vocab_file):
            self.vocab_file = os.path.join(self.data_dir, "vocab.txt")
        task_names_str = ",".join(
            kwargs["task_names"] if "task_names" in kwargs else self.task_names)
        self.init_checkpoint = None if self.debug else pretrained_model_dir
        self.model_dir = os.path.join(pretrained_model_dir, "finetuning_models",
                                      task_names_str + "_model")
        results_dir = os.path.join(pretrained_model_dir, "results")
        self.results_txt = os.path.join(results_dir,
                                        task_names_str + "_results.txt")
        self.results_pkl = os.path.join(results_dir,
                                        task_names_str + "_results.pkl")
        qa_topdir = os.path.join(results_dir, task_names_str + "_qa")
        self.qa_eval_file = os.path.join(qa_topdir, "{:}_eval.json").format
        self.qa_preds_file = os.path.join(qa_topdir, "{:}_preds.json").format
        self.qa_na_file = os.path.join(qa_topdir, "{:}_null_odds.json").format
        self.preprocessed_data_dir = os.path.join(
            pretrained_model_dir, "finetuning_tfrecords",
            task_names_str + "_tfrecords" + ("-debug" if self.debug else ""))
        self.test_predictions = os.path.join(
            pretrained_model_dir, "test_predictions",
            "{:}_{:}_{:}_predictions.pkl").format

        # update defaults with passed-in hyperparameters
        self.update(kwargs)

        # default hyperparameters for single-task models
        if len(self.task_names) == 1:
            task_name = self.task_names[0]
            if task_name == "rte" or task_name == "sts":
                self.num_train_epochs = 10.0
            elif "squad" in task_name or "qa" in task_name:
                self.max_seq_length = 512
                self.num_train_epochs = 2.0
                self.write_distill_outputs = False
                self.write_test_outputs = False
            elif task_name == "chunk":
                self.max_seq_length = 256
            else:
                self.num_train_epochs = 3.0

        # default hyperparameters for different model sizes
        if self.model_size == "large":
            self.learning_rate = 5e-5
            self.layerwise_lr_decay = 0.9
        elif self.model_size == "small":
            self.embedding_size = 128

        # debug-mode settings
        if self.debug:
            self.save_checkpoints_steps = 1000000
            self.use_tfrecords_if_existing = False
            self.num_trials = 1
            self.iterations_per_loop = 1
            self.train_batch_size = 32
            self.num_train_epochs = 3.0
            self.log_examples = True

        # passed-in-arguments override (for example) debug-mode defaults
        self.update(kwargs)

    def update(self, kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                raise ValueError("Unknown hparam " + k)
            self.__dict__[k] = v
