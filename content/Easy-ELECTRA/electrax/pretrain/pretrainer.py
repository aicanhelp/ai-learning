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

from electrax.bert import modeling
from electrax.bert import optimization
from electrax.pretrain import pretrain_data
from electrax.pretrain import pretrain_helpers
from electrax.util import training_utils
from electrax.util import tfutils
from electrax.configuration import PretrainingConfig


class TransformerBuilder:
    """BERT model. Although the training algorithm is different, the transformer
       model for ELECTRA is the same as BERT's.

       Example usage:

       ```python
       # Already been converted into WordPiece token ids
       input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
       input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
       token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

       config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
         num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

       model = modeling.BertModel(config=config, is_training=True,
         input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

       label_embeddings = tf.get_variable(...)
       pooled_output = model.get_pooled_output()
       logits = tf.matmul(pooled_output, label_embeddings)
       ...
       ```
       """

    def __init__(self, config: PretrainingConfig):
        self.config = config

    def build(self, inputs: pretrain_data.Inputs, is_training,
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


class GeneratorInput:
    """
        This'is the input from tfrecords.
        For how to create the tfrecords, please refer to 'pretrain_data.get_input_fn'.
        'pretrain_data.get_input_fn' as a input_fn of tf, it'is used in PretrainingTrainer for training
    """

    def __init__(self, config: PretrainingConfig, features):
        self.config = config
        self.features = features
        self.masked_inputs = self._create_masked_inputs()

    def _create_masked_inputs(self):
        return pretrain_helpers.mask(
            self.config, pretrain_data.features_to_inputs(self.features), self.config.mask_prob)


class GeneratorModel:
    def __init__(self, config: PretrainingConfig,
                 generatorInput: GeneratorInput,
                 transformerBuilder: TransformerBuilder,
                 is_training):
        self.config = config
        self.is_training = is_training
        self.generatorInput = generatorInput
        self.transformerBuilder = transformerBuilder
        self.model = self._create_generator()

    def _create_generator(self):
        if self.config.electra_objective and self.config.untied_generator:
            return self.transformerBuilder.build(
                self.generatorInput.masked_inputs, self.is_training,
                bert_config=self._get_generator_config(self.config, self.config.bert_config),
                embedding_size=(None if self.config.untied_generator_embeddings
                                else self.config.embedding_size),
                untied_embeddings=self.config.untied_generator_embeddings,
                name="generator")
        else:
            return self.transformerBuilder.build(
                self.generatorInput.masked_inputs, self.is_training, embedding_size=self.config.embedding_size)

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


class GeneratorOutput:
    def __init__(self, config: PretrainingConfig, generatorModel: GeneratorModel):
        self.config = config
        self.generatorModel = generatorModel
        self.mlm_output = self._get_masked_lm_output(self.generatorModel.generatorInput.masked_inputs,
                                                     self.generatorModel.model)

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


class DiscriminatorInput:
    """
    The discriminator is trained to distinguish tokens in the data from tokens that have been replaced by generator samples.
    More specifically, we create acorrupted example x_corrupt by replacing the masked-out tokens with generator samples and
    train the discriminator to predict which tokens in x_corrupt match the original input x."""

    def __init__(self, config: PretrainingConfig, generatorInput: GeneratorInput, generatorOutput: GeneratorOutput):
        self.config = config
        self.fake_data = self._get_fake_data(
            generatorInput.masked_inputs,
            generatorOutput.mlm_output.logits)

    def _get_fake_data(self, inputs, mlm_logits):
        """Sample from the generator to create corrupted input."""
        inputs = pretrain_helpers.unmask(inputs)

        # whether exclude the tokens preidicted by generator
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


class DiscriminatorModel:
    def __init__(self, config: PretrainingConfig, transformerBuilder: TransformerBuilder,
                 discriminatorInput: DiscriminatorInput, is_training):
        self.config = config
        self.is_training = is_training
        self.transformerBuilder = transformerBuilder
        self.discriminatorInput = discriminatorInput
        self.model = self._create_discriminator()

    def _create_discriminator(self):
        if self.config.electra_objective:
            return self.transformerBuilder.build(
                self.discriminatorInput.fake_data.inputs, self.is_training,
                reuse=not self.config.untied_generator,
                embedding_size=self.config.embedding_size)


class DiscriminatorOutput:
    def __init__(self, config: PretrainingConfig, discriminatorModel: DiscriminatorModel):
        self.config = config
        self.discriminatorModel = discriminatorModel
        self.output = self._get_discriminator_output()

    def _get_discriminator_output(self):
        inputs = self.discriminatorModel.discriminatorInput.fake_data.inputs
        labels = self.discriminatorModel.discriminatorInput.fake_data.is_fake_tokens
        """Discriminator binary classifier."""
        with tf.variable_scope("discriminator_predictions"):
            hidden = tf.layers.dense(
                self.discriminatorModel.model.get_sequence_output(),
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


class PretrainedLoss:
    def __init__(self, config: PretrainingConfig, generatorOutput: GeneratorOutput,
                 discriminatorOutput: DiscriminatorOutput):
        self.total_loss = config.gen_weight * generatorOutput.mlm_output.loss
        if config.electra_objective:
            self.total_loss += config.disc_weight * discriminatorOutput.output.loss


class PretrainingModelEvalMetrics:
    def __init__(self, config: PretrainingConfig,
                 generatorInput: GeneratorInput,
                 generatorOutput: GeneratorOutput,
                 discriminatorInput: DiscriminatorInput,
                 discriminatorOutput: DiscriminatorOutput):
        self.config = config
        self.generatorInput = generatorInput
        self.generatorOutput = generatorOutput
        self.discriminatorInput = discriminatorInput
        self.discriminatorOutput = discriminatorOutput
        self.eval_fn_inputs = self._eval_fn_inputs()
        self.eval_metrics = (self._eval_fn_inputs())

    def _eval_fn_inputs(self):
        masked_inputs = self.generatorInput.masked_inputs
        mlm_output = self.generatorOutput.mlm_output
        eval_fn_inputs = {
            "input_ids": masked_inputs.input_ids,
            "masked_lm_preds": mlm_output.preds,
            "mlm_loss": mlm_output.per_example_loss,
            "masked_lm_ids": masked_inputs.masked_lm_ids,
            "masked_lm_weights": masked_inputs.masked_lm_weights,
            "input_mask": masked_inputs.input_mask
        }
        if self.config.electra_objective:
            disc_output = self.discriminatorOutput.output
            eval_fn_inputs.update({
                "disc_loss": disc_output.per_example_loss,
                "disc_labels": disc_output.labels,
                "disc_probs": disc_output.probs,
                "disc_preds": disc_output.preds,
                "sampled_tokids": tf.argmax(self.discriminatorInput.fake_data.sampled_tokens, -1,
                                            output_type=tf.int32)
            })
        eval_fn_keys = eval_fn_inputs.keys()
        eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]
        return eval_fn_inputs, eval_fn_values


class PretrainedModel():
    def __init__(self, config: PretrainingConfig, features, is_training):
        ##Bert Model buider
        self.transformerBuilder = TransformerBuilder(config)

        ## Generator module
        self.generatorInput = GeneratorInput(config, features)
        self.generatorModel = GeneratorModel(config, self.generatorInput, self.transformerBuilder, is_training)
        self.generatorOutput = GeneratorOutput(config, self.generatorModel)

        ##Discriminator module
        self.discriminatorInput = DiscriminatorInput(config, self.generatorInput, self.generatorOutput)
        self.discriminatorModel = DiscriminatorModel(config, self.transformerBuilder, self.discriminatorInput,
                                                     is_training)
        self.discriminatorOutput = DiscriminatorOutput(config, self.discriminatorModel)

        ##Loss
        self.loss = PretrainedLoss(config, self.generatorOutput, self.discriminatorOutput)

        ##Metrics
        self.eval_metrics = PretrainingModelEvalMetrics(config, self.generatorInput, self.generatorOutput,
                                                        self.discriminatorInput, self.discriminatorOutput)


class PretrainingRunner():
    def __init__(self, config: PretrainingConfig):
        self.config = config
        self._init()
        self.tpu_config, self.run_config = self._run_config()

    def _init(self):
        if self.config.do_train == self.config.do_eval:
            raise ValueError("Exactly one of `do_train` or `do_eval` must be True.")
        if self.config.debug:
            tfutils.rmkdir(self.config.model_dir)
        tfutils.heading("Config:")
        tfutils.log_config(self.config)

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

    def run(self, build_model_fn, runAction):
        model_fn = build_model_fn(self.config)
        estimator = self._create_estimator(model_fn)
        return runAction(estimator)

    def create_model(self, config: PretrainingConfig, features, mode):
        model = PretrainedModel(config, features,
                                mode == tf.estimator.ModeKeys.TRAIN)
        tfutils.log("Model is built!")
        return model


class PretrainingTrainer():
    def __init__(self, runner: PretrainingRunner):
        self.runner = runner

    def _build_model_fn(self, config):
        def model_fn(features, labels, mode, params):
            """Build the model for training."""
            model = self.runner.create_model(config, features, mode)

            train_op = optimization.create_optimizer(
                model.loss.total_loss,
                config.learning_rate,
                config.num_train_steps,
                weight_decay_rate=config.weight_decay_rate,
                use_tpu=config.use_tpu,
                warmup_steps=config.num_warmup_steps,
                lr_decay_power=config.lr_decay_power
            )
            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=model.loss.total_loss,
                train_op=train_op,
                training_hooks=[training_utils.ETAHook(
                    {} if config.use_tpu else dict(loss=model.loss.total_loss),
                    config.num_train_steps, config.iterations_per_loop,
                    config.use_tpu)]
            )

            return output_spec

        return model_fn

    def _train(self, estimator):
        tfutils.heading("Running training")
        estimator.train(input_fn=pretrain_data.get_input_fn(self.runner.config, True),
                        max_steps=self.runner.config.num_train_steps)

    def start(self):
        self.runner.run(self._build_model_fn, self._train)


class PretrainingEvaluator():
    def __init__(self, runner: PretrainingRunner):
        self.runner = runner

    def _build_model_fn(self, config):
        def model_fn(features, labels, mode, params):
            """Build the model for training."""
            model = self.runner.create_model(config, features, mode)

            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=model.loss.total_loss,
                eval_metrics=model.eval_metrics,
                evaluation_hooks=[training_utils.ETAHook(
                    {} if config.use_tpu else dict(loss=model.loss.total_loss),
                    config.num_eval_steps, config.iterations_per_loop,
                    config.use_tpu, is_training=False)])

            return output_spec

        return model_fn

    def _eval(self, estimator):
        tfutils.heading("Running evaluation")
        result = estimator.evaluate(
            input_fn=pretrain_data.get_input_fn(self.runner.config, False),
            steps=self.runner.config.num_eval_steps)
        for key in sorted(result.keys()):
            tfutils.log("  {:} = {:}".format(key, str(result[key])))
        return result

    def start(self):
        self.runner.run(self._build_model_fn, self._eval)


def train_one_step(config: PretrainingConfig):
    """Builds an ELECTRA model an trains it for one step; useful for debugging."""
    train_input_fn = pretrain_data.get_input_fn(config, True)
    features = tf.data.make_one_shot_iterator(train_input_fn(dict(
        batch_size=config.train_batch_size))).get_next()
    model = PretrainedModel(config, features, True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tfutils.log(sess.run(model.loss.total_loss))


def start(config: PretrainingConfig):
    runner: PretrainingRunner = PretrainingRunner(config)
    if config.do_train:
        return PretrainingTrainer(runner).start()
    if config.do_eval:
        return PretrainingEvaluator(runner).start()


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
        hparams = tfutils.load_json(args.hparams)
    else:
        hparams = json.loads(args.hparams)
    tf.logging.set_verbosity(tf.logging.ERROR)
    start(PretrainingConfig(
        args.model_name, args.data_dir, **hparams))


if __name__ == "__main__":
    main()
