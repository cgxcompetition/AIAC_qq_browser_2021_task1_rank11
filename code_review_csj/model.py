# -*- coding: utf-8 -*-
# @Author : chenshaojie
# @File : model.py
# @Software: cross modal model
import tensorflow as tf
from tensorflow.python.keras.models import Model
from transformers import TFBertModel, create_optimizer, BertConfig, LxmertConfig
from transformers.modeling_tf_bert import TFBertEncoder
from transformers.modeling_tf_lxmert import TFLxmertXLayer, TFLxmertLayer
from transformers.modeling_tf_utils import get_initializer


class NeXtVLAD(tf.keras.layers.Layer):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.expansion = expansion
        self.groups = groups

        self.new_feature_size = expansion * feature_size // groups
        self.expand_dense = tf.keras.layers.Dense(self.expansion * self.feature_size)
        # for group attention
        self.attention_dense = tf.keras.layers.Dense(self.groups, activation=tf.nn.sigmoid)
        # self.activation_bn = tf.keras.layers.BatchNormalization()

        # for cluster weights
        self.cluster_dense1 = tf.keras.layers.Dense(self.groups * self.cluster_size, activation=None, use_bias=False)
        # self.cluster_dense2 = tf.keras.layers.Dense(self.cluster_size, activation=None, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(rate=dropout, seed=1)
        self.fc = tf.keras.layers.Dense(output_size, activation=None)

    def build(self, input_shape):
        self.cluster_weights2 = self.add_weight(name="cluster_weights2",
                                                shape=(1, self.new_feature_size, self.cluster_size),
                                                initializer=tf.keras.initializers.glorot_normal, trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        image_embeddings, mask = inputs
        _, num_segments, _ = image_embeddings.shape
        if mask is not None:  # in case num of images is less than num_segments
            images_mask = tf.sequence_mask(mask, maxlen=num_segments)
            images_mask = tf.cast(tf.expand_dims(images_mask, -1), tf.float32)
            image_embeddings = tf.multiply(image_embeddings, images_mask)
        inputs = self.expand_dense(image_embeddings)
        attention = self.attention_dense(inputs)

        attention = tf.reshape(attention, [-1, num_segments * self.groups, 1])
        reshaped_input = tf.reshape(inputs, [-1, self.expansion * self.feature_size])

        activation = self.cluster_dense1(reshaped_input)
        # activation = self.activation_bn(activation)
        activation = tf.reshape(activation, [-1, num_segments * self.groups, self.cluster_size])
        activation = tf.nn.softmax(activation, axis=-1)  # shape: batch_size * (max_frame*groups) * cluster_size
        activation = tf.multiply(activation, attention)  # shape: batch_size * (max_frame*groups) * cluster_size

        a_sum = tf.reduce_sum(activation, -2, keepdims=True)  # shape: batch_size * 1 * cluster_size
        a = tf.multiply(a_sum, self.cluster_weights2)  # shape: batch_size * new_feature_size * cluster_size
        activation = tf.transpose(activation, perm=[0, 2, 1])  # shape: batch_size * cluster_size * (max_frame*groups)

        reshaped_input = tf.reshape(inputs, [-1, num_segments * self.groups, self.new_feature_size])

        vlad = tf.matmul(activation, reshaped_input)  # shape: batch_size * cluster_size * new_feature_size
        vlad = tf.transpose(vlad, perm=[0, 2, 1])  # shape: batch_size * new_feature_size * cluster_size
        vlad = tf.subtract(vlad, a)
        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.new_feature_size])

        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(tf.keras.layers.Layer):
    def __init__(self, channels, ratio=8, **kwargs):
        super(SENet, self).__init__(**kwargs)
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(channels // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False),
            tf.keras.layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        ])

    def call(self, inputs, **kwargs):
        se = self.fc(inputs)
        outputs = tf.math.multiply(inputs, se)
        return outputs


class ConcatDenseSE(tf.keras.layers.Layer):
    """Fusion using Concate + Dense + SENet"""

    def __init__(self, hidden_size, se_ratio, **kwargs):
        super().__init__(**kwargs)
        self.fusion = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fusion_dropout = tf.keras.layers.Dropout(0.2)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def call(self, inputs, **kwargs):
        embeddings = tf.concat(inputs, axis=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding


class Extractmask(tf.keras.layers.Layer):
    """extract vision mask"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(768, activation='relu')

    def call(self, inputs, **kwargs):
        image_embeddings, mask = inputs
        frame_num = tf.reshape(mask, [-1])
        _, num_segments, _ = image_embeddings.shape
        frame_mask = tf.sequence_mask(frame_num, maxlen=num_segments)
        frame_mask = tf.cast(frame_mask, tf.int32)
        frame = self.fc(image_embeddings)
        return frame, frame_mask


class TFBertPooler(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            256,
            activation='tanh',
            name="dense",
        )

    def call(self, hidden_states):


        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)

        return pooled_output


class TFLxmertEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # self.visn_fc = TFLxmertVisualFeatureEncoder(config, name="visn_fc")

        # Number of layers
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = [TFLxmertLayer(config, name="layer_._{}".format(i)) for i in range(self.num_l_layers)]
        self.x_layers = [TFLxmertXLayer(config, name="x_layers_._{}".format(i)) for i in range(self.num_x_layers)]
        self.r_layers = [TFLxmertLayer(config, name="r_layers_._{}".format(i)) for i in range(self.num_r_layers)]
        self.config = config
        self.dense = tf.keras.layers.Dense(
            512,
            kernel_initializer=get_initializer(config.initializer_range),
            activation='tanh',
            name="dense"
        )
        self.maxpool = tf.keras.layers.GlobalMaxPool1D()
        self.avgpool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(
            config.emb_hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation='relu',
        )

    def call(
        self,
        lang_feats=None,
        lang_attention_mask=None,
        visual_feats=None,
        visual_attention_mask=None,
        output_attentions=None,
        training=False,
    ):
        vision_hidden_states = ()
        language_hidden_states = ()
        vision_attentions = () if output_attentions or self.config.output_attentions else None
        language_attentions = () if output_attentions or self.config.output_attentions else None
        cross_encoder_attentions = () if output_attentions or self.config.output_attentions else None

        # visual_feats = self.visn_fc([visual_feats, visual_pos], training=training)
        # Run language layers
        for layer_module in self.layer:
            l_outputs = layer_module(lang_feats, lang_attention_mask, output_attentions, training=training)
            lang_feats = l_outputs[0]
            language_hidden_states = language_hidden_states + (lang_feats,)
            if language_attentions is not None:
                language_attentions = language_attentions + (l_outputs[1],)

        # Run relational layers
        for layer_module in self.r_layers:
            v_outputs = layer_module(
                visual_feats,
                visual_attention_mask,
                output_attentions,
                training=training,
            )
            visual_feats = v_outputs[0]
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            if vision_attentions is not None:
                vision_attentions = vision_attentions + (v_outputs[1],)

        # Run cross-modality layers
        for layer_module in self.x_layers:
            x_outputs = layer_module(
                lang_feats,
                lang_attention_mask,
                visual_feats,
                visual_attention_mask,
                output_attentions,
                training=training,
            )
            lang_feats, visual_feats = x_outputs[:2]
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            language_hidden_states = language_hidden_states + (lang_feats,)
            if cross_encoder_attentions is not None:
                cross_encoder_attentions = cross_encoder_attentions + (x_outputs[2],)

        visual_encoder_outputs = (
            vision_hidden_states,
            vision_attentions if self.config.output_attentions else None,
        )
        lang_encoder_outputs = (
            language_hidden_states,
            language_attentions if self.config.output_attentions else None,
        )
        vision_hidden_states = visual_encoder_outputs[0]
        language_hidden_states = lang_encoder_outputs[0]

        visual_output = vision_hidden_states[-1]
        lang_output = language_hidden_states[-1]

        lang_output = self.avgpool(lang_output)
        visual_output = self.avgpool(visual_output)
        all_output = tf.concat([visual_output, lang_output], axis=1)
        all_output = self.dense1(all_output)
        return all_output


class CGXMultiModal(Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = TFBertModel.from_pretrained(config.bert_dir)
        self.nextvlad = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
                                 output_size=config.vlad_hidden_size, dropout=config.dropout)
        self.fusion = ConcatDenseSE(config.hidden_size, config.se_ratio)
        self.extract_mask = Extractmask()
        self.num_labels = config.num_labels
        self.classifier = tf.keras.layers.Dense(self.num_labels, activation='sigmoid')

        self.bert_optimizer, self.bert_lr = create_optimizer(init_lr=config.bert_lr,
                                                             num_train_steps=config.bert_total_steps,
                                                             num_warmup_steps=config.bert_warmup_steps)
        self.optimizer, self.lr = create_optimizer(init_lr=config.lr,
                                                   num_train_steps=config.total_steps,
                                                   num_warmup_steps=config.warmup_steps)
        self.bert_variables, self.num_bert, self.normal_variables, self.all_variables = None, None, None, None
        cfg1 = LxmertConfig()
        cfg1.l_layers = 0
        cfg1.x_layers = 6
        cfg1.r_layers = 4
        cfg1.initializer_range = 0.02
        cfg1.output_attentions = False
        cfg1.output_hidden_states = False
        cfg1.emb_hidden_size = 256
        self.cross_modal = TFLxmertEncoder(config=cfg1)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2, seed=1)
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2, seed=1)
        self.TFBertPool = TFBertPooler()

    def call(self, inputs, **kwargs):
        query_bert_embedding = self.bert([inputs['input_query_ids'], inputs['query_mask']])

        frame, frame_mask = self.extract_mask([inputs['query_frames'], inputs['query_num_frames']])
        # frame_mask
        video_attention_mask = frame_mask[:, tf.newaxis, tf.newaxis, :]
        video_attention_mask = tf.cast(video_attention_mask, tf.float32)
        video_attention_mask = (1.0 - video_attention_mask) * -10000.0

        extended_attention_mask = inputs['query_mask'][:, tf.newaxis, tf.newaxis, :]
        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        query_final_embedding = self.cross_modal(query_bert_embedding[0], extended_attention_mask, frame, video_attention_mask, False)

        # 另一个塔
        candidate_bert_embedding = self.bert([inputs['input_candidate_ids'], inputs['candidate_mask']])

        candidate_frame, candidate_frame_mask = self.extract_mask([inputs['candidate_frames'], inputs['candidate_num_frames']])
        # frame_mask
        video_attention_mask1 = candidate_frame_mask[:, tf.newaxis, tf.newaxis, :]
        video_attention_mask1 = tf.cast(video_attention_mask1, tf.float32)
        video_attention_mask1 = (1.0 - video_attention_mask1) * -10000.0

        extended_attention_mask1 = inputs['candidate_mask'][:, tf.newaxis, tf.newaxis, :]
        extended_attention_mask1 = tf.cast(extended_attention_mask1, tf.float32)
        extended_attention_mask1 = (1.0 - extended_attention_mask1) * -10000.0

        candidate_final_embedding = self.cross_modal(candidate_bert_embedding[0], extended_attention_mask1, candidate_frame, video_attention_mask1, False)

        query_final_embedding = self.dense1(query_final_embedding)
        candidate_final_embedding = self.dense1(candidate_final_embedding)

        query_norm_embedding = tf.nn.l2_normalize(query_final_embedding, axis=1)
        candidate_norm_embedding = tf.nn.l2_normalize(candidate_final_embedding, axis=1)
        predictions = tf.math.multiply(query_norm_embedding, candidate_norm_embedding)
        predictions = tf.reduce_sum(predictions, axis=1)

        query_label_predictions = self.classifier(query_final_embedding)
        candidate_label_predictions = self.classifier(candidate_final_embedding)

        return predictions, query_final_embedding, candidate_final_embedding, query_label_predictions, candidate_label_predictions

    def get_variables(self):
        if not self.all_variables:  # is None, not initialized
            self.bert_variables = self.bert.trainable_variables
            self.num_bert = len(self.bert_variables)
            self.normal_variables = self.extract_mask.trainable_variables + self.cross_modal.trainable_variables + \
                                    self.dense1.trainable_variables + self.classifier.trainable_variables
            self.all_variables = self.bert_variables + self.normal_variables
        return self.all_variables

    def optimize(self, gradients):
        bert_gradients = gradients[:self.num_bert]
        self.bert_optimizer.apply_gradients(zip(bert_gradients, self.bert_variables))
        normal_gradients = gradients[self.num_bert:]
        self.optimizer.apply_gradients(zip(normal_gradients, self.normal_variables))
