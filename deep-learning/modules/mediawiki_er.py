import tensorflow as tf
from tqdm import tqdm_notebook
from collections import namedtuple

def reset_tf(sess = None, log_device_placement = False):
    if sess:
        sess.close()
    tf.reset_default_graph()
    tf.set_random_seed(0)
    return tf.InteractiveSession(config = tf.ConfigProto(log_device_placement = log_device_placement))

class BaseModel:
    def __init__(self, hp):
        self._hp = hp
        self._build_model()
    
    def _parse_example(self, example_proto):
        features = {
            'page_id': tf.FixedLenFeature([1], tf.int64),
            'para_id': tf.FixedLenFeature([1], tf.int64),
            'sentence_id': tf.FixedLenFeature([1], tf.int64),
            'inputs': tf.VarLenFeature(tf.int64),
            'targets': tf.VarLenFeature(tf.int64)
        }

        parsed = tf.parse_single_example(example_proto, features)

        def convert_and_pad(sparse_tensor):
            result = tf.sparse_tensor_to_dense(sparse_tensor)
            # TODO: properly ignore elements which are too large (right now we just clip)
            result = result[:self._hp.max_sequence_length]
            result = tf.pad(result, [[0, self._hp.max_sequence_length - tf.shape(result)[0]]])
            return result

        return (parsed['page_id'],
                parsed['para_id'],
                parsed['sentence_id'],
                convert_and_pad(parsed['inputs']),
                tf.shape(parsed['inputs'])[0],
                convert_and_pad(parsed['targets']))
    
    def _build_data_pipeline(self):
        with tf.variable_scope('dataset'):
            self._dataset_filenames = tf.placeholder(tf.string, shape = [None])

            dataset = tf.data.TFRecordDataset(self._dataset_filenames)
            dataset = dataset.map(self._parse_example,
                                  num_parallel_calls = self._hp.pipeline_num_parallel_calls)
            dataset = dataset.shuffle(self._hp.pipeline_shuffle_size)
            dataset = dataset.prefetch(self._hp.pipeline_prefetch_size)
            dataset = dataset.batch(self._hp.pipeline_batch_size)

            self._dataset_iterator = dataset.make_initializable_iterator()
            (input_page_ids,
             input_para_ids,
             input_sentence_ids,
             input_sequences, 
             input_lengths, 
             target_sequences) = self._dataset_iterator.get_next()
            
            self._input_page_ids = tf.placeholder_with_default(input_page_ids,
                                                               shape = [None, 1],
                                                               name = 'input_page_ids')
            self._input_para_ids = tf.placeholder_with_default(input_para_ids,
                                                               shape = [None, 1],
                                                               name = 'input_para_ids')
            self._input_sentence_ids = tf.placeholder_with_default(input_sentence_ids,
                                                                   shape = [None, 1],
                                                                   name = 'input_sentence_ids')
            self._input_sequences = tf.placeholder_with_default(input_sequences,
                                                                shape = [None, self._hp.max_sequence_length],
                                                                name = 'input_sequences')
            self._input_lengths = tf.placeholder_with_default(input_lengths,
                                                              shape = [None],
                                                              name = 'input_lengths')
            self._target_sequences = tf.placeholder_with_default(target_sequences,
                                                                 shape = [None, self._hp.max_sequence_length],
                                                                 name = 'target_sequences')

            self._input_positions = tf.range(self._hp.max_sequence_length, dtype = tf.int64)
            self._input_positions = tf.tile(self._input_positions, [tf.shape(self._input_sequences)[0]])
            self._input_positions = tf.reshape(self._input_positions, 
                                               (tf.shape(self._input_sequences)[0], self._hp.max_sequence_length), 
                                               name = 'input_positions')
    
    def _build_error_model(self):
        with tf.variable_scope('errors'):
            sequence_mask = tf.sequence_mask(self._input_lengths,
                                             self._hp.max_sequence_length,
                                             dtype = tf.bool)

            self._output_sequences = tf.nn.softmax(self._output_logits)
            self._output_sequences = tf.argmax(self._output_sequences, axis = -1)
            self._output_sequences *= tf.cast(sequence_mask, tf.int64)

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self._target_sequences,
                                                                    logits = self._output_logits)
            losses *= tf.cast(sequence_mask, tf.float32)

            self._total_loss = tf.reduce_sum(losses)
            self._total_input_length = tf.reduce_sum(self._input_lengths)
            self._mean_loss  = self._total_loss / tf.cast(self._total_input_length, tf.float32)

            self._true_positives = tf.reduce_sum(self._output_sequences * self._target_sequences)
            self._false_positives = tf.reduce_sum(tf.maximum(self._output_sequences - self._target_sequences, 0))
            self._false_negatives = tf.reduce_sum(tf.maximum(self._target_sequences - self._output_sequences, 0))

    def _build_training_model(self):
        self._global_step = tf.Variable(0, name = 'global_step', trainable = False)
        optimizer = tf.train.AdamOptimizer(learning_rate = self._hp.learning_rate)
        self._train_op = optimizer.minimize(self._mean_loss, global_step = self._global_step)
        
    def _build_prediction_model(self):
        self._output_logits = self._build_prediction_model_internal()
        
    def _build_model(self):
        self._is_training = tf.placeholder(tf.bool)
        
        self._build_data_pipeline()
        self._build_prediction_model()
        self._build_error_model()
        self._build_training_model()
        
    def dump_statistics(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print('parameters for "%s": %d' % (variable.name, variable_parameters))
            total_parameters += variable_parameters
        print('total parameters: %d' % total_parameters)
    
    def evaluate_dataset(self,
                         sess,
                         dataset_filename,
                         options = None,
                         run_metadata = None,
                         header = 'results',
                         train = False,
                         show_progress = True):
        cum_loss = 0
        cum_input_length = 0
        cum_true_positives = 0
        cum_false_positives = 0
        cum_false_negatives = 0

        sess.run(self._dataset_iterator.initializer, feed_dict={
            self._dataset_filenames: [dataset_filename]
        })

        if show_progress:
            progress = tqdm_notebook()

        while True:
            try:
                (_,
                 curr_loss, 
                 curr_input_length, 
                 curr_true_positives,
                 curr_false_positives,
                 curr_false_negatives) = sess.run((self._train_op if train else [],
                                                   self._total_loss,
                                                   self._total_input_length,
                                                   self._true_positives,
                                                   self._false_positives,
                                                   self._false_negatives),
                                                  feed_dict = { self._is_training: train },
                                                  options = options,
                                                  run_metadata = run_metadata)
            except tf.errors.OutOfRangeError:
                break

            if show_progress:
                progress.update(curr_input_length)

            cum_loss += curr_loss
            cum_input_length += curr_input_length
            cum_true_positives += curr_true_positives
            cum_false_positives += curr_false_positives
            cum_false_negatives += curr_false_negatives

        if show_progress:
            progress.close()

        precision = cum_true_positives / (cum_true_positives + cum_false_positives)
        recall = cum_true_positives / (cum_true_positives + cum_false_negatives)
        F1 = 2 * (precision * recall) / (precision + recall)

        print('%s (%d): loss=%g, precision=%g, recall=%g, F1=%g' % (header,
                                                                    tf.train.global_step(sess, self._global_step),
                                                                    cum_loss/cum_input_length, 
                                                                    precision, 
                                                                    recall, 
                                                                    F1))