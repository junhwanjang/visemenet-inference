import tensorflow as tf
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import logfbank, mfcc, ssc
from postprocess import postprocess_model_outputs


class VisemeRegressor(object):
    def __init__(self, pb_filepath):
        # Load forzen graph
        self.pb_filepath = pb_filepath
        self.graph = self._load_graph(self.pb_filepath)
        
        # Define Hpyer-params
        ## Sampling
        self.fps = 25
        self.mfcc_win_step_per_frame = 1
        self.up_sample_rate = 4
        self.win_length = 0.025
        self.winstep = 1.0 / self.fps / self.mfcc_win_step_per_frame / self.up_sample_rate
        self.window_size = 24

        ## Num Signal features
        self.num_mfcc = 13
        self.num_logfbank = 26
        self.num_ssc = 26
        self.num_total_features = 65

        ## Model Params
        self.n_steps = 8
        self.n_input = int(self.num_total_features * self.mfcc_win_step_per_frame * self.window_size / self.n_steps)
        self.n_landmark = 76
        self.n_face_id = 76
        self.n_phoneme = 21
        self.n_maya_params = 22
    
    def predict_outputs(self, wav_file_path, mean_std_csv_path='./saved_params/wav_mean_std.csv', close_face_txt_path='./saved_params/maya_close_face.txt'):
        # Define Input
        ## Preprocess wav file
        concat_feat = self._preprocess_wav(
            wav_file_path=wav_file_path, is_debug=False
        )
        normalized_feat = self._normalize_input(
            concat_features=concat_feat, mean_std_csv_path=mean_std_csv_path
        )
        target_wav_idxs = self._get_padded_indexes(
            normalized_feat=normalized_feat, window_size=self.window_size
        )
        ## Prepare model input
        batch_size = concat_feat.shape[0] # Num Frames
        batch_x, batch_x_face_id = self._prepare_model_input(
            normalized_feat=normalized_feat, 
            target_wav_idxs=target_wav_idxs,
            batch_size=batch_size, 
            close_face_txt_path=close_face_txt_path
        )
        
        # Predict Outputs
        ## Input nodes
        x = self.graph.get_tensor_by_name('input/Placeholder_1:0')
        x_face_id = self.graph.get_tensor_by_name('input/Placeholder_2:0')
        phase = self.graph.get_tensor_by_name('input/phase:0')
        dropout = self.graph.get_tensor_by_name('net1_shared_rnn/Placeholder:0')

        ## Output nodes
        v_cls = self.graph.get_tensor_by_name('net2_output/add_1:0')
        v_reg = self.graph.get_tensor_by_name('net2_output/add_4:0')
        jali = self.graph.get_tensor_by_name('net2_output/add_6:0')

        with tf.compat.v1.Session(graph=self.graph) as sess:
            pred_v_cls, pred_v_reg, pred_jali = sess.run(
                [v_cls, v_reg, jali],
                feed_dict={
                    x: batch_x, 
                    x_face_id: batch_x_face_id,
                    dropout: 0, phase: 0
                    }
            )
            pred_v_cls = self.sigmoid(pred_v_cls)

        # Postprocess Outputs - Smoothing and Clip based on the pre-calculated thresholds
        cls_output = np.concatenate([pred_jali, pred_v_cls], axis=1)
        reg_output = np.concatenate([pred_jali, pred_v_reg], axis=1)

        viseme_outputs = postprocess_model_outputs(
            reg_output=reg_output, cls_output=cls_output
        )
        
        return viseme_outputs

    def _prepare_model_input(self, normalized_feat, target_wav_idxs, batch_size, close_face_txt_path):
        batch_x = np.zeros((batch_size, self.n_steps, self.n_input))
        batch_x_face_id = np.zeros((batch_size, self.n_face_id))
        # batch_x_pose = np.zeros((batch_size, 3))
        # batch_y_landmark = np.zeros((batch_size, self.n_landmark))
        # batch_y_phoneme = np.zeros((batch_size, self.n_phoneme))
        # batch_y_lipS = np.zeros((batch_size, 1))
        # batch_y_maya_param = np.zeros((batch_size, self.n_maya_params))

        for i in range(0, batch_size):
            batch_x[i] = normalized_feat[target_wav_idxs[i]].reshape((-1, self.n_steps, self.n_input))

        close_face = np.loadtxt(close_face_txt_path)
        batch_x_face_id = np.tile(close_face, (batch_size, 1))
        
        return batch_x, batch_x_face_id

    def _get_padded_indexes(self, normalized_feat, window_size):
        # Get Padded indexes based on the given window size
        num_frames = normalized_feat.shape[0]
        wav_idxs = [i for i in range(0, num_frames)]
        
        half_win_size = window_size // 2
        pad_head = [0 for _ in range(half_win_size)]
        pad_tail = [wav_idxs[-1] for _ in range(half_win_size)]
        padded_idxs = np.array(pad_head + wav_idxs + pad_tail)
        
        target_wav_idxs = np.zeros(shape=(num_frames, window_size)).astype(int)
        for i in range(0, num_frames):
            target_wav_idxs[i] = padded_idxs[i:i+window_size].reshape(-1, window_size)

        return target_wav_idxs

    def _normalize_input(self, concat_features, mean_std_csv_path):
        # Normalize input using the pre-calculated mean, std values
        num_features = self.num_mfcc + self.num_logfbank + self.num_ssc
        
        mean_std = np.loadtxt(mean_std_csv_path)
        mean_vals = mean_std[:num_features]
        std_vals = mean_std[num_features:]

        normalized_feat = (concat_features - mean_vals) / std_vals
        
        return normalized_feat

    def _preprocess_wav(self, wav_file_path, is_debug=False):
        sample_rate, signal = wav.read(wav_file_path)

        if (signal.ndim > 1):
            signal = signal[:, 0]

        # Get concatentated features
        ## 1. mfcc_features
        mfcc_feat = mfcc(
            signal, numcep=self.num_mfcc,
            samplerate=sample_rate,
            winlen=self.win_length, winstep=self.winstep
        )

        ## 2. logfbank_features
        logfbank_feat = logfbank(
            signal, nfilt=self.num_logfbank,
            samplerate=sample_rate,
            winlen=self.win_length, winstep=self.winstep
        )

        ## 3. ssc_features
        ssc_feat = ssc(
            signal, nfilt=self.num_ssc,
            samplerate=sample_rate,
            winlen=self.win_length, winstep=self.winstep
        )

        concat_features = np.concatenate(
            [mfcc_feat, logfbank_feat, ssc_feat], axis=1
        )

        target_frames = int(concat_features.shape[0] / self.mfcc_win_step_per_frame / self.up_sample_rate)
        mfcc_lines = concat_features[:target_frames * self.mfcc_win_step_per_frame * self.up_sample_rate]

        if is_debug:
            print("Sample Rate: {}".format(sample_rate))
            print("Signal Shape: {}".format(signal.shape))
            print("")
            print("Collect Features")
            print("[mfcc feat shape]: {}".format(mfcc_feat.shape))
            print("[logfbank feat shape]: {}".format(logfbank_feat.shape))
            print("[ssc feat shape]: {}".format(ssc_feat.shape))
            print("--> Concat Features Shape: {}".format(concat_features.shape))
        
        return mfcc_lines

    def _load_graph(self, pb_filepath):
        with tf.io.gfile.GFile(pb_filepath, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        
        for op in graph.get_operations():
            if op.type == 'Placeholder':
                print(op.name)

        return graph

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
