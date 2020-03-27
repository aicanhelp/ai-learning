# coding=utf-8
# refactored from google's Electra implementation
"""Pre-trains an ELECTRA model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json

import tensorflow.compat.v1 as tf

import configure_pretraining
from model import modeling
from model import optimization
from pretrain import pretrain_data
from pretrain import pretrain_helpers
from util import training_utils
from util import utils
from configure_pretraining import PretrainingConfig


class PretrainedModelBase():
    def __int__(self, config: PretrainingConfig):
        self.config = config

    def _build_transformer(self, inputs: pretrain_data.Inputs, is_training,
                           bert_config=None, name="electra", reuse=False, **kwargs):
        """Build a transformer encoder network."""
        if bert_config is None:
            bert_config = self.config.bert_config
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            return modeling.BertModel(
                bert_config=bert_config,
                is_training=is_training,
                input_ids=inputs.input_ids,
                input_mask=inputs.input_mask,
                token_type_ids=inputs.segment_ids,
                use_one_hot_embeddings=self.config.use_tpu,
                scope=name,
                **kwargs)


class PretrainedModelGenerator(PretrainedModelBase):
    def __init__(self, config: PretrainingConfig, features, is_training):
        super(self).__int__(config)
        self.features = features
        self.is_training = is_training

        self.masked_inputs = self._create_masked_inputs()
        self.generator, self.mlm_output = self._create_generator_mlm_output()

    def _create_masked_inputs(self):
        return pretrain_helpers.mask(
            self.config, pretrain_data.features_to_inputs(self.features), self.config.mask_prob)

    def _create_generator_mlm_output(self):
        generator, mlm_output = None, None
        if self.config.uniform_generator:
            mlm_output = self._get_masked_lm_output(self.masked_inputs, None)
        elif self.config.electra_objective and self.config.untied_generator:
            generator = self._build_transformer(
                self.masked_inputs, self.is_training,
                bert_config=self._get_generator_config(self.config, self.config.bert_config),
                embedding_size=(None if self.config.untied_generator_embeddings
                                else self.config.embedding_size),
                untied_embeddings=self.config.untied_generator_embeddings,
                name="generator")
            mlm_output = self._get_masked_lm_output(self.masked_inputs, generator)
        else:
            generator = self._build_transformer(
                self.masked_inputs, self.is_training, embedding_size=self.config.embedding_size)
            mlm_output = self._get_masked_lm_output(self.masked_inputs, generator)

        return generator, mlm_output

    def _get_generator_config(self, config: PretrainingConfig,
                              bert_config: modeling.BertConfig):
        """Get model config for the generator network."""

        gen_config = modeling.BertConfig.from_dict(bert_config.to_dict())
        gen_config.hidden_size = int(round(
            bert_config.hidden_size * config.generator_hidden_size))
        gen_config.num_hidden_layers = int(round(
            bert_config.num_hidden_layers * config.generator_layers))
        gen_config.intermediate_size = 4 * gen_config.hidden_size
        gen_config.num_attention_heads = max(1, gen_config.hidden_size // 64)
        return gen_config

    def _get_masked_lm_output(self, inputs: pretrain_data.Inputs, model):
        """Masked language modeling softmax layer."""
        masked_lm_weights = inputs.masked_lm_weights
        with tf.variable_scope("generator_predictions"):
            if self.config.uniform_generator:
                logits = tf.zeros(self.config.bert_config.vocab_size)
                logits_tiled = tf.zeros(
                    modeling.get_shape_list(inputs.masked_lm_ids) +
                    [self.config.bert_config.vocab_size])
                logits_tiled += tf.reshape(logits, [1, 1, self.config.bert_config.vocab_size])
                logits = logits_tiled
            else:
                relevant_hidden = pretrain_helpers.gather_positions(
                    model.get_sequence_output(), inputs.masked_lm_positions)
                hidden = tf.layers.dense(
                    relevant_hidden,
                    units=modeling.get_shape_list(model.get_embedding_table())[-1],
                    activation=modeling.get_activation(self.config.bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        self.config.bert_config.initializer_range))
                hidden = modeling.layer_norm(hidden)
                output_bias = tf.get_variable(
                    "output_bias",
                    shape=[self.config.bert_config.vocab_size],
                    initializer=tf.zeros_initializer())
                logits = tf.matmul(hidden, model.get_embedding_table(),
                                   transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)

            oh_labels = tf.one_hot(
                inputs.masked_lm_ids, depth=self.config.bert_config.vocab_size,
                dtype=tf.float32)

            probs = tf.nn.softmax(logits)
            log_probs = tf.nn.log_softmax(logits)
            label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)

            numerator = tf.reduce_sum(inputs.masked_lm_weights * label_log_probs)
            denominator = tf.reduce_sum(masked_lm_weights) + 1e-6
            loss = numerator / denominator
            preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

            MLMOutput = collections.namedtuple(
                "MLMOutput", ["logits", "probs", "loss", "per_example_loss", "preds"])
            return MLMOutput(
                logits=logits, probs=probs, per_example_loss=label_log_probs,
                loss=loss, preds=preds)


class PretrainedModelDiscriminator(PretrainedModelBase):
    def __init__(self, config: PretrainingConfig, generator: PretrainedModelGenerator, is_training):
        super(self).__int__(config)
        self.generator = generator
        self.is_training = is_training
        self.fake_data = self._get_fake_data(self.generator.masked_inputs, self.generator.mlm_output.logits)
        self.total_loss = config.gen_weight * self.generator.mlm_output.loss
        self.discriminator, self.disc_output = self._create_discriminator()

    def _create_discriminator(self):
        discriminator, disc_output = None, None
        if self.config.electra_objective:
            discriminator = self._build_transformer(
                self.fake_data.inputs, self.is_training, reuse=not self.config.untied_generator,
                embedding_size=self.config.embedding_size)
            disc_output = self._get_discriminator_output(
                self.fake_data.inputs, discriminator, self.fake_data.is_fake_tokens)
            self.total_loss += self.config.disc_weight * disc_output.loss
        return discriminator, disc_output

    def _get_fake_data(self, inputs, mlm_logits):
        """Sample from the generator to create corrupted input."""
        inputs = pretrain_helpers.unmask(inputs)
        disallow = tf.one_hot(
            inputs.masked_lm_ids, depth=self.config.bert_config.vocab_size,
            dtype=tf.float32) if self.config.disallow_correct else None
        sampled_tokens = tf.stop_gradient(pretrain_helpers.sample_from_softmax(
            mlm_logits / self.config.temperature, disallow=disallow))
        sampled_tokids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
        updated_input_ids, masked = pretrain_helpers.scatter_update(
            inputs.input_ids, sampled_tokids, inputs.masked_lm_positions)
        labels = masked * (1 - tf.cast(
            tf.equal(updated_input_ids, inputs.input_ids), tf.int32))
        updated_inputs = pretrain_data.get_updated_inputs(
            inputs, input_ids=updated_input_ids)
        FakedData = collections.namedtuple("FakedData", [
            "inputs", "is_fake_tokens", "sampled_tokens"])
        return FakedData(inputs=updated_inputs, is_fake_tokens=labels,
                         sampled_tokens=sampled_tokens)

    def _get_discriminator_output(self, inputs, discriminator, labels):
        """Discriminator binary classifier."""
        with tf.variable_scope("discriminator_predictions"):
            hidden = tf.layers.dense(
                discriminator.get_sequence_output(),
                units=self.config.bert_config.hidden_size,
                activation=modeling.get_activation(self.config.bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    self.config.bert_config.initializer_range))
            logits = tf.squeeze(tf.layers.dense(hidden, units=1), -1)
            weights = tf.cast(inputs.input_mask, tf.float32)
            labelsf = tf.cast(labels, tf.float32)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labelsf) * weights
            per_example_loss = (tf.reduce_sum(losses, axis=-1) /
                                (1e-6 + tf.reduce_sum(weights, axis=-1)))
            loss = tf.reduce_sum(losses) / (1e-6 + tf.reduce_sum(weights))
            probs = tf.nn.sigmoid(logits)
            preds = tf.cast(tf.round((tf.sign(logits) + 1) / 2), tf.int32)
            DiscOutput = collections.namedtuple(
                "DiscOutput", ["loss", "per_example_loss", "probs", "preds",
                               "labels"])
            return DiscOutput(
                loss=loss, per_example_loss=per_example_loss, probs=probs,
                preds=preds, labels=labels,
            )


class PretrainingModelEvalMetrics:
    def __init__(self, config: PretrainingConfig,
                 generator: PretrainedModelGenerator,
                 discriminator: PretrainedModelDiscriminator):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.eval_fn_inputs = self._eval_fn_inputs()
        self.eval_metrics = (self._eval_fn_inputs())

    def _eval_fn_inputs(self):
        eval_fn_inputs = {
            "input_ids": self.generator.masked_inputs.input_ids,
            "masked_lm_preds": self.generator.mlm_output.preds,
            "mlm_loss": self.generator.mlm_output.per_example_loss,
            "masked_lm_ids": self.generator.masked_inputs.masked_lm_ids,
            "masked_lm_weights": self.generator.masked_inputs.masked_lm_weights,
            "input_mask": self.generator.masked_inputs.input_mask
        }
        if self.config.electra_objective:
            eval_fn_inputs.update({
                "disc_loss": self.discriminator.disc_output.per_example_loss,
                "disc_labels": self.discriminator.disc_output.labels,
                "disc_probs": self.discriminator.disc_output.probs,
                "disc_preds": self.discriminator.disc_output.preds,
                "sampled_tokids": tf.argmax(self.discriminator.fake_data.sampled_tokens, -1,
                                            output_type=tf.int32)
            })
        eval_fn_keys = eval_fn_inputs.keys()
        eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]
        return eval_fn_inputs, eval_fn_values


class PretrainingModel(object):
    """Transformer pre-training using the replaced-token-detection task."""

    def __init__(self, config: PretrainingConfig,
                 features, is_training):
        # Set up model config
        self._config = config.build()

        self.generator = PretrainedModelGenerator(config, features, is_training)
        self.discriminator = PretrainedModelDiscriminator(config, self.generator, is_training)
        self.eval_metrics = PretrainingModelEvalMetrics(config, self.generator, self.discriminator)


class PretrainingRunner():
    def __init__(self, config: PretrainingConfig):
        self.config = config
        self._init()
        self.tpu_config, self.run_config = self._run_config()

    def _init(self):
        if self.config.do_train == self.config.do_eval:
            raise ValueError("Exactly one of `do_train` or `do_eval` must be True.")
        if self.config.debug:
            utils.rmkdir(self.config.model_dir)
        utils.heading("Config:")
        utils.log_config(self.config)

    def _run_config(self):
        is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
        tpu_cluster_resolver = None
        if self.config.use_tpu and self.config.tpu_name:
            tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                self.config.tpu_name, zone=self.config.tpu_zone, project=self.config.gcp_project)

        tpu_config = tf.estimator.tpu.TPUConfig(
            iterations_per_loop=self.config.iterations_per_loop,
            num_shards=(self.config.num_tpu_cores if self.config.do_train else
                        self.config.num_tpu_cores),
            tpu_job_name=self.config.tpu_job_name,
            per_host_input_for_training=is_per_host)

        run_config = tf.estimator.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=self.config.model_dir,
            save_checkpoints_steps=self.config.save_checkpoints_steps,
            tpu_config=tpu_config)
        return tpu_config, run_config

    def _create_estimator(self, model_fn):
        return tf.estimator.tpu.TPUEstimator(
            use_tpu=self.config.use_tpu,
            model_fn=model_fn,
            config=self.run_config,
            train_batch_size=self.config.train_batch_size,
            eval_batch_size=self.config.eval_batch_size)

    def run(self):
        model_fn = self._build_model_fn(self.config)
        estimator = self._create_estimator(model_fn)
        return self.doRun(estimator)

    def create_model(self, config: PretrainingConfig, features, mode):
        model = PretrainingModel(config, features,
                                 mode == tf.estimator.ModeKeys.TRAIN)
        utils.log("Model is built!")
        return model

    def _build_model_fn(self, config: PretrainingConfig):
        pass

    def doRun(self, estimator):
        pass


class PretrainingTrainer(PretrainingRunner):
    def __init(self, config: PretrainingConfig):
        super(self).__init__(config)

    def _build_model_fn(self, config):
        def model_fn(features, labels, mode, params):
            """Build the model for training."""
            model = self.create_model(config, features, mode)

            train_op = optimization.create_optimizer(
                model.discriminator.total_loss,
                config.learning_rate,
                config.num_train_steps,
                weight_decay_rate=config.weight_decay_rate,
                use_tpu=config.use_tpu,
                warmup_steps=config.num_warmup_steps,
                lr_decay_power=config.lr_decay_power
            )
            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=model.discriminator.total_loss,
                train_op=train_op,
                training_hooks=[training_utils.ETAHook(
                    {} if config.use_tpu else dict(loss=model.discriminator.total_loss),
                    config.num_train_steps, config.iterations_per_loop,
                    config.use_tpu)]
            )

            return output_spec

        return model_fn

    def doRun(self, estimator):
        utils.heading("Running training")
        estimator.train(input_fn=pretrain_data.get_input_fn(self.config, True),
                        max_steps=self.config.num_train_steps)


class PretrainingEvaluator(PretrainingRunner):
    def __init(self, config: PretrainingConfig):
        super(self).__init__(config)

    def _build_model_fn(self, config):
        def model_fn(features, labels, mode, params):
            """Build the model for training."""
            model = self.create_model(config, features, mode)

            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=model.discriminator.total_loss,
                eval_metrics=model.eval_metrics,
                evaluation_hooks=[training_utils.ETAHook(
                    {} if config.use_tpu else dict(loss=model.discriminator.total_loss),
                    config.num_eval_steps, config.iterations_per_loop,
                    config.use_tpu, is_training=False)])

            return output_spec

        return model_fn

    def doRun(self, estimator):
        utils.heading("Running evaluation")
        result = estimator.evaluate(
            input_fn=pretrain_data.get_input_fn(self.config, False),
            steps=self.config.num_eval_steps)
        for key in sorted(result.keys()):
            utils.log("  {:} = {:}".format(key, str(result[key])))
        return result


def train_one_step(config: PretrainingConfig):
    """Builds an ELECTRA model an trains it for one step; useful for debugging."""
    train_input_fn = pretrain_data.get_input_fn(config, True)
    features = tf.data.make_one_shot_iterator(train_input_fn(dict(
        batch_size=config.train_batch_size))).get_next()
    model = PretrainingModel(config, features, True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        utils.log(sess.run(model.discriminator.total_loss))


def start(config: PretrainingConfig):
    if config.do_train:
        return PretrainingTrainer(config).run()
    if config.do_eval:
        return PretrainingEvaluator(config).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Location of data files (model weights, etc).")
    parser.add_argument("--model-name", required=True,
                        help="The name of the model being fine-tuned.")
    parser.add_argument("--hparams", default="{}",
                        help="JSON dict of model hyperparameters.")
    args = parser.parse_args()
    if args.hparams.endswith(".json"):
        hparams = utils.load_json(args.hparams)
    else:
        hparams = json.loads(args.hparams)
    tf.logging.set_verbosity(tf.logging.ERROR)
    start(configure_pretraining.PretrainingConfig(
        args.model_name, args.data_dir, **hparams))


if __name__ == "__main__":
    main()
