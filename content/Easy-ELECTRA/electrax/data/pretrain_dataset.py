"""
Writes out text data as tfrecords that ELECTRA can be pre-trained on.
Refactored from Google's implementation
"""

import random
import tarfile
import time

from electrax.bert import tokenization
from electrax.configuration import *
from aiharness.executors import Executor


class ElectraExampleBuilder:
    '''
        ElectraExample includes two segments: first_segment_ids and second_segment_ids.
        In small chance, only have the first_segment as in classification tasks, and the second segment is none
        The serialized result is:
        (1) input_ids=[cls_id]+first_segment_ids+[sep_id]+second_ids+[sep_id]
            especially: some examples for classification, input_ids=[cls_id]+first_segment_ids+[sep_id]
        (2) input_mask=[1] * len(input_ids)
        (3) segment_ids=[0]+[0]*len(first_segment_ids)+[0]+[1]*len(second_segment_ids)+[1]
        Finally, if the length of input_ids, input_mask, segment_ids is less than max length, then pad them with 0

        For a documents, half of sentences are put into first_segment randomly, and another half of sentences are
        put into second_segment randomly.
    '''

    def __init__(self, max_length, cls_id, sep_id, cls_rate=0.1, cls_segment_length=100000):
        self.max_length, self.cls_id, self.sep_id = max_length, cls_id, sep_id
        self.cls_rate, self.cls_segment_length = cls_rate, cls_segment_length

    def build(self, sentences_ids, target_length):
        """Creates a pre-training example from the current list of sentences."""
        # small chance to only have one segment as in classification tasks
        if random.random() < 0.1:
            first_segment_target_length = 100000
        else:
            # -3 due to not yet having [CLS]/[SEP] tokens in the input text
            first_segment_target_length = (target_length - 3) // 2

        first_segment = []
        second_segment = []
        for sentence in sentences_ids:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (first_segment or
                    len(first_segment) + len(sentence) < first_segment_target_length or
                    (second_segment and
                     len(first_segment) < first_segment_target_length and
                     random.random() < 0.5)):
                first_segment += sentence
            else:
                second_segment += sentence

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        first_segment = first_segment[:self.max_length - 2]
        second_segment = second_segment[:max(0, self.max_length -
                                             len(first_segment) - 3)]
        return self.__serialize(first_segment, second_segment)

    def __serialize(self, first_segment_ids, second_segment_ids):
        input_ids = [self.cls_id] + first_segment_ids + [self.sep_id]
        segment_ids = [0] * len(input_ids)
        if second_segment_ids:
            input_ids += second_segment_ids + [self.sep_id]
            segment_ids += [1] * (len(second_segment_ids) + 1)
        input_mask = [1] * len(input_ids)
        input_ids += [0] * (self.max_length - len(input_ids))
        input_mask += [0] * (self.max_length - len(input_mask))
        segment_ids += [0] * (self.max_length - len(segment_ids))
        return input_ids, input_mask, segment_ids


class TextExampleParser(object):
    def __init__(self, vocab_file, max_length, do_lower_case):
        self._current_sentences = []
        self._current_length = 0
        self._max_length = max_length
        self._target_length = max_length

        self._tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case)
        self.example_builder = ElectraExampleBuilder(self._max_length,
                                                     self._tokenizer.vocab[CLS],
                                                     self._tokenizer.vocab[SEP],
                                                     )

    def on_example(self, onExample):
        self.onExample = onExample
        return self

    def _is_doc_end(self, line):
        # empty lines separate docs
        return (not line) and self._current_length != 0

    def add_line(self, line):
        """Adds a line of text to the current example being built."""
        line = line.strip().replace("\n", " ")
        if self._is_doc_end(line):
            self._create_example()

        bert_tokens = self._tokenizer.tokenize(line)
        bert_tokids = self._tokenizer.convert_tokens_to_ids(bert_tokens)
        self._current_sentences.append(bert_tokids)
        self._current_length += len(bert_tokids)

        if self._current_length >= self._target_length:
            self._create_example()

    def _create_example(self):
        # small chance for random-length instead of max_length-length example
        if random.random() < 0.05:
            self._target_length = random.randint(5, self._max_length)
        else:
            self._target_length = self._max_length

        self.onExample(self.example_builder.build(self._current_sentences, self._target_length))


class TFExampleWriter(object):
    def __init__(self, job_id, num_jobs, output_dir, num_out_files=1000):
        self.job_id = job_id
        self.num_dout_files = num_out_files
        self._writers = []
        for i in range(num_out_files):
            if i % num_jobs == job_id:
                output_fname = os.path.join(
                    output_dir, "pretrain_data.tfrecord-{:}-of-{:}".format(
                        i, num_out_files))
                self._writers.append(tf.io.TFRecordWriter(output_fname))
        self.n_written = 0

    def write(self, input_ids, input_mask, segment_ids):
        example = self.__create_tf_example(input_ids, input_mask, segment_ids)
        self._writers[self.n_written % len(self._writers)].write(
            example.SerializeToString())
        self.n_written += 1

    def __create_tf_example(self, input_ids, input_mask, segment_ids):
        return tf.train.Example(features=tf.train.Features(feature={
            "input_ids": self.__create_int_feature(input_ids),
            "input_mask": self.__create_int_feature(input_mask),
            "segment_ids": self.__create_int_feature(segment_ids)
        }))

    def __create_int_feature(self, values):
        feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return feature

    def finish(self):
        for writer in self._writers:
            writer.close()


class TextExampleReader(object):
    def __init__(self, blanks_separate_docs):
        self._blanks_separate_docs = blanks_separate_docs

    def on_line(self, onLine):
        self._onLine = onLine
        return self

    def read(self, input_file):
        with tf.io.gfile.GFile(input_file) as f:
            for line in f:
                line = line.strip()
                if line or self._blanks_separate_docs:
                    self._onLine(line)
            self._onLine("")


class TFPretrainExamplesGenerator():
    def __init__(self, job_id, config: DataGeneratorConfig):
        self.job_id = job_id
        self.config = config
        self.tfExampleWriter = TFExampleWriter(job_id, config.num_process, config.output_dir)

        self.textExampleParser = TextExampleParser(config.vocab_file, config.max_seq_length, config.do_lower_case)
        self.textExampleParser.on_example(self.tfExampleWriter.write)

        self.textExampleReader = TextExampleReader(config.blanks_separate_docs)
        self.textExampleReader.on_line(self.textExampleParser.add_line)

    def _log(self, start_time, file_no, fnames):
        elapsed = time.time() - start_time
        data_log.info("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s, ETA: {:}s, "
                      "{:} examples written".format(
            file_no, len(fnames), 100.0 * file_no / len(fnames), int(elapsed),
            int((len(fnames) - file_no) / (file_no / elapsed)),
            self.tfExampleWriter.n_written))

    def _exstract_file(self, fname):
        if self.config.flatten:
            return [self.config.corpus_dir, fname]
        job_tmp_dir = os.path.join(self.config.output_dir, "tmp", "job_" + str(self.job_id))
        tfutils.rmkdir(job_tmp_dir)
        with tarfile.open(os.path.join(self.config.corpus_dir, fname)) as f:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, job_tmp_dir)
        extracted_files = tf.io.gfile.listdir(job_tmp_dir)
        random.shuffle(extracted_files)
        return extracted_files

    def run(self):
        data_log.info("Writing tf examples")
        fnames = sorted(tf.io.gfile.listdir(self.config.corpus_dir))
        fnames = [f for (i, f) in enumerate(fnames)
                  if i % self.config.num_process == self.job_id]
        random.shuffle(fnames)
        start_time = time.time()
        for file_no, fname in enumerate(fnames):
            if file_no > 0 and file_no % 10 == 0:
                self._log(start_time, file_no, fnames)
            dir, extracted_files = self._exstract_file(fname)
            for txt_fname in extracted_files:
                self.textExampleReader.read(os.path.join(dir, txt_fname))
        self.tfExampleWriter.finish()
        data_log.info("Done!")


class DataGeneratorRunner:
    def __init__(self):
        self.config: DataGeneratorConfig = get_data_config()

    def generator_fn(self, job_id):
        TFPretrainExamplesGenerator(job_id, self.config).run()

    def run(self):
        if self.config.num_process == 1:
            TFPretrainExamplesGenerator(0, self.config).run()
            return

        Executor(self.generator_fn).get()
