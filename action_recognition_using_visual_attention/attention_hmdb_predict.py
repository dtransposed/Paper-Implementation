import tensorflow as tf
from models import ALSTM, ConvALSTM
from data_readers.hmdb_reader import HMDBDataReader
from attention_inference_engine_hmdb import SequentialInference
from utils import AttentionTrainer, f1_score, plotter, use_only_specified_gpu, build_attention_network
import os
import yaml
import shutil
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score


def predict(config_file):
    with open(config_file) as stream:
        config_data = yaml.safe_load(stream)

    use_only_specified_gpu(config_data['data_creation_parameters']['GPU_id'])

    shutil.rmtree(config_data['prediction_save_dir'])
    os.mkdir(config_data['prediction_save_dir'])

    reader = HMDBDataReader(dataset_directory=config_data['dataset_save_dir'],
                            batch_size=config_data['training_parameters']['batch_size'],
                            sequence_len=config_data['data_creation_parameters']['sequence_len'],
                            base_type=config_data['testing_parameters']['base_type'])

    _, test_ds, _ = reader.get_datasets_sequence()

    if bool(config_data['testing_parameters']['display_test_data']):
        print("Displaying sequence...")
        reader.display_sequences_test()

    base_model_type = config_data['testing_parameters']['base_type']

    if base_model_type == 'VGG':
        base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        feature_map_size = 7
        image_input_size = 224
        filter_no = 512

    else:
        raise ValueError("No proper base model chosen for training!")

    ##### CHOOSING THE ARCHITECTURE #####

    units_first_lstm = config_data['training_parameters']['lstm_parameters'][0]

    if config_data['testing_parameters']['model_type'] == 'ALSTM':

        print('Testing ALSTM')

        classifier = ALSTM(base_model=base_model,
                           use_dropout=bool(config_data['training_parameters']['use_dropout']),
                           train_base=bool(config_data['training_parameters']['train_base']),
                           units_first_lstm=units_first_lstm,
                           no_classes=config_data['training_parameters']['no_classes'],
                           feature_map_size=feature_map_size)

    elif config_data['testing_parameters']['model_type'] == 'ConvALSTM':

        print('Testing ConvALSTM')

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
                            config_data['testing_parameters']['model_type'],
                            feature_map_size,
                            image_input_size,
                            filter_no)

    classifier.load_weights(os.path.join(config_data['model_save_dir'],
                                         config_data['testing_parameters']['model_type'],
                                         config_data['testing_parameters']['current_model_to_load']))

    inference_engine = SequentialInference(config_data)

    all_labels = []
    all_predictions = []

    for (batch, (image_test, labels_test)) in enumerate(test_ds):
        if image_test.shape[0] != config_data['training_parameters']['batch_size']:
            print('Invalid batch size, skipped...')
        else:
            all_video_predictions = np.zeros((config_data['data_creation_parameters']['sequence_len'],
                                              image_test.shape[0],
                                              config_data['training_parameters']['no_classes']))

            if config_data['testing_parameters']['model_type'] == "ConvALSTM":
                hidden_state = classifier.get_initial_hidden_state(image_test.shape[0])
                for i in range(0, image_test.shape[1]):
                    input_image = image_test[:, i, :, :, :]
                    predictions, hidden_state, attention_map = classifier(input_image, hidden_state)
                    all_video_predictions[i, :, :] = predictions
                    inference_engine.save_image(input_image, attention_map, predictions, labels_test, i)

                all_video_predictions = np.argmax(all_video_predictions, axis=2)
                label_mode = stats.mode(all_video_predictions)[0]
                all_labels = all_labels + list(labels_test.numpy())
                all_predictions = all_predictions + list(label_mode[0])




            elif config_data['testing_parameters']['model_type'] == "ALSTM":
                batch_feature_cube_sequence = classifier.get_batch_feature_cube_sequence(image_test)
                hidden_state, cell_state = classifier.reset_hidden_and_cell_state(batch_feature_cube_sequence=
                                                                                  batch_feature_cube_sequence)
                classifier.lstm1.initial_state = cell_state
                for i in range(0, image_test.shape[1]):
                    input_image = image_test[:, i, :, :, :]
                    predictions, hidden_state, attention_map = classifier(input_image, hidden_state)
                    all_video_predictions[i, :, :] = predictions
                    inference_engine.save_image(input_image, attention_map, predictions, labels_test, i)

                all_video_predictions = np.argmax(all_video_predictions, axis=2)
                label_mode = stats.mode(all_video_predictions)[0]
                all_labels = all_labels + list(labels_test.numpy())
                all_predictions = all_predictions + list(label_mode[0])

    print("Accuracy on test data: {}".format(accuracy_score(all_predictions, all_labels)))


if __name__ == "__main__":
    predict('config.yaml')
