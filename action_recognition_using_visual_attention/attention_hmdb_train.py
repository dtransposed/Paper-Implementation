import tensorflow as tf
from utils import AttentionTrainer, f1_score, plotter, use_only_specified_gpu, build_attention_network
from models import ALSTM, ConvALSTM
from data_readers.hmdb_reader import HMDBDataReader
import os
import yaml
import io
import datetime


def train(config_file):

    with open(config_file) as stream:
        config_data = yaml.safe_load(stream)

    use_only_specified_gpu(config_data['data_creation_parameters']['GPU_id'])

    base_model_type = config_data['training_parameters']['base_type']

    print("Reading dataset...")
    reader = HMDBDataReader(dataset_directory=config_data['dataset_save_dir'],
                            batch_size= config_data['training_parameters']['batch_size'],
                            sequence_len=config_data['data_creation_parameters']['sequence_len'],
                            base_type = base_model_type)

    train_ds, _, val_ds = reader.get_datasets_sequence()

    if bool(config_data['data_creation_parameters']['display_training_data']):
        print("Displaying sequence...")
        reader.display_sequences_train()

    ##### CHOOSING THE BASE MODEL #####

    if base_model_type == 'VGG':
        base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        feature_map_size = 7
        image_input_size = 224
        filter_no = 512

    else:
        raise ValueError("No proper base model chosen for training!")

    ##### CHOOSING THE ARCHITECTURE #####

    units_first_lstm = config_data['training_parameters']['lstm_parameters'][0]

    if config_data['training_parameters']['model_type'] == 'ALSTM':

        print('Training ALSTM')

        classifier = ALSTM(base_model=base_model,
                           use_dropout=bool(config_data['training_parameters']['use_dropout']),
                           train_base=bool(config_data['training_parameters']['train_base']),
                           units_first_lstm=units_first_lstm,
                           no_classes=config_data['training_parameters']['no_classes'],
                           feature_map_size=feature_map_size)

    elif config_data['training_parameters']['model_type'] == 'ConvALSTM':

        print('Training ConvALSTM')

        classifier = ConvALSTM(base_model=base_model,
                           use_dropout=bool(config_data['training_parameters']['use_dropout']),
                           train_base=bool(config_data['training_parameters']['train_base']),
                           units_first_lstm=units_first_lstm,
                           no_classes=config_data['training_parameters']['no_classes'],
                           feature_map_size=feature_map_size)
    else:
        raise ValueError("No proper model chosen for training.")

    build_attention_network(classifier,
                            config_data['training_parameters']['batch_size'],
                            config_data['data_creation_parameters']['sequence_len'],
                            units_first_lstm,
                            config_data['training_parameters']['model_type'],
                            feature_map_size,
                            image_input_size,
                            filter_no)

    ##### START TRAINING #####

    trainer = AttentionTrainer(
                      network = classifier,
                      network_name = config_data['training_parameters']['model_type'],
                      optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
                      loss_object = tf.keras.losses.SparseCategoricalCrossentropy(),
                      loss_object_attention_pen = tf.keras.losses.MeanSquaredError(),
                      train_loss = tf.keras.metrics.Mean(name='train_loss'),
                      train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy'),
                      train_precision = tf.keras.metrics.Precision(name='train_precision'),
                      train_recall = tf.keras.metrics.Recall(name='train_recall'),
                      val_loss=tf.keras.metrics.Mean(name='val_loss'),
                      val_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy'),
                      val_precision=tf.keras.metrics.Precision(name='val_precision'),
                      val_recall=tf.keras.metrics.Recall(name='val_recall'),
                      test_loss=tf.keras.metrics.Mean(name='test_loss'),
                      test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy'),
                      test_precision=tf.keras.metrics.Precision(name='test_precision'),
                      test_recall=tf.keras.metrics.Recall(name='test_recall'),
                      batch_size = config_data['training_parameters']['batch_size'],
                      penalty_coeff = config_data['training_parameters']['penalty_coeff'],
                      weight_decay=config_data['training_parameters']['weight_decay']
                      )

    training_data = []

    for epoch in range(config_data['training_parameters']['epochs']):
        trainer.train_loss.reset_states()
        trainer.train_accuracy.reset_states()
        trainer.train_precision.reset_states()
        trainer.train_recall.reset_states()
        trainer.val_loss.reset_states()
        trainer.val_accuracy.reset_states()
        trainer.val_precision.reset_states()
        trainer.val_recall.reset_states()

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        train_log_dir = os.path.join(config_data['model_save_dir'], 'logs/gradient_tape/') + current_time + '/train'
        val_log_dir = os.path.join(config_data['model_save_dir'], 'logs/gradient_tape/') + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        print('Training...')
        for (batch, (image, labels)) in enumerate(train_ds):
            if image.shape[0] != config_data['training_parameters']['batch_size']:
                print('Invalid batch size, skipped...')
            else:
                trainer.train_step(image, labels)

            with train_summary_writer.as_default():
                tf.summary.scalar('train loss', trainer.train_loss.result(), step=batch)
                tf.summary.scalar('train accuracy', trainer.train_accuracy.result(), step=batch)

        print('Validating...')
        for (batch, (image_val, labels_val)) in enumerate(val_ds):
            if image_val.shape[0] != config_data['training_parameters']['batch_size']:
                print('Invalid batch size, skipped...')
            else:
                trainer.val_step(image_val, labels_val)

            with val_summary_writer.as_default():
                tf.summary.scalar('val loss', trainer.val_loss.result(), step=batch)
                tf.summary.scalar('val accuracy', trainer.val_accuracy.result(), step=batch)

        epoch_summary = (epoch + 1,
                         trainer.train_loss.result(),
                         trainer.train_accuracy.result(),
                         f1_score(trainer.train_precision.result(), trainer.train_recall.result()),
                         trainer.val_loss.result(),
                         trainer.val_accuracy.result(),
                         f1_score(trainer.val_precision.result(), trainer.val_recall.result()))

        training_data.append(epoch_summary)

        template = 'Epoch {}, Loss: {}, Accuracy:{}, F1 Score: {}, Val Loss: {}, Val Acc: {}, Val F1 Score: {}'

        print(template.format(epoch_summary[0],
                              epoch_summary[1],
                              epoch_summary[2],
                              epoch_summary[3],
                              epoch_summary[4],
                              epoch_summary[5],
                              epoch_summary[6],))

        model_savedir = os.path.join(config_data['model_save_dir'], config_data['training_parameters']['model_type'])
        if not os.path.exists(model_savedir):
            os.mkdir(model_savedir)

        plotter(training_data, model_savedir)
        with io.open(os.path.join(model_savedir, '{}'.format(config_file)), 'w', encoding='utf8') as outfile:
            yaml.dump(config_data, outfile, default_flow_style=False, allow_unicode=True)

        trainer.save_weights(epoch, model_savedir)
        print("Weights have been saved. Epoch {} done!".format(epoch_summary[0]))

if __name__ == "__main__":
    train("config.yaml")









