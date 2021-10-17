import tensorflow as tf
from tensorflow.python.keras.models import Model
from transformers import TFBertModel, create_optimizer


class NeXtVLAD(tf.keras.layers.Layer):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2,name='nextVLAD'):
        super().__init__(name=name)
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


class CGXMultiModal(Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = TFBertModel.from_pretrained(config.bert_dir)
        self.nextvlad = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
                                 output_size=config.vlad_hidden_size, dropout=config.dropout)
        self.nextvlad_diff = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
                                 output_size=config.vlad_hidden_size, dropout=config.dropout,name='nextVLADdiff')
        self.fusion = ConcatDenseSE(config.hidden_size, config.se_ratio)

        self.num_labels = config.num_labels
        self.classifier = tf.keras.layers.Dense(self.num_labels, activation='sigmoid')

        self.bert_optimizer, self.bert_lr = create_optimizer(init_lr=config.bert_lr,
                                                             num_train_steps=config.bert_total_steps,
                                                             num_warmup_steps=config.bert_warmup_steps)
        self.optimizer, self.lr = create_optimizer(init_lr=config.lr,
                                                   num_train_steps=config.total_steps,
                                                   num_warmup_steps=config.warmup_steps)
        self.bert_variables, self.num_bert, self.normal_variables, self.all_variables = None, None, None, None
    def tf_diff_axis_0(self, frame):
        return tf.concat([frame[:,1:,:], tf.reshape(frame[:,0,:],(-1,1,1536))],axis=1)[0]-frame
    def frame_forward(self,inputs,num_frame_key,nextvlad_model,frame_key,is_diff=False):
        frame_num = tf.reshape(inputs[num_frame_key],[-1])
        vision_embedding=nextvlad_model([self.tf_diff_axis_0(inputs[frame_key]) if is_diff else inputs[frame_key],frame_num])
        vision_embedding=vision_embedding*tf.cast(tf.expand_dims(frame_num,-1)>0,tf.float32)
        return vision_embedding
    def fusion_bert_nextvlad(self,inputs,input_ids_key,mask_key,num_frame_key,frame_key):
        bert_embedding=self.bert([inputs[input_ids_key],inputs[mask_key]])[1]
        vision_embedding=self.frame_forward(inputs,num_frame_key=num_frame_key,nextvlad_model=self.nextvlad,frame_key=frame_key)
        vision_embedding_diff=self.frame_forward(inputs,num_frame_key=num_frame_key,nextvlad_model=self.nextvlad_diff,frame_key=frame_key,is_diff=True)
        # vision_embedding_diff=self.frame_forward(inputs,num_frame_key=num_frame_key,nextvlad_model=self.nextvlad_diff,frame_key=frame_key)
        final_embedding=self.fusion([vision_embedding,bert_embedding,vision_embedding_diff])
        return final_embedding
    def call(self, inputs, **kwargs):
        query_final_embedding = self.fusion_bert_nextvlad(
            inputs,
            input_ids_key='input_query_ids',
            mask_key='query_mask',
            num_frame_key='query_num_frames',
            frame_key='query_frames'
        )
        candidate_final_embedding = self.fusion_bert_nextvlad(
            inputs,
            input_ids_key='input_candidate_ids',
            mask_key='candidate_mask',
            num_frame_key='candidate_num_frames',
            frame_key='candidate_frames'
        )
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
            self.normal_variables = self.nextvlad.trainable_variables +  \
            self.nextvlad_diff.trainable_variables+ \
            self.fusion.trainable_variables + \
            self.classifier.trainable_variables
            self.all_variables = self.bert_variables + self.normal_variables
        return self.all_variables

    def optimize(self, gradients):
        bert_gradients = gradients[:self.num_bert]
        self.bert_optimizer.apply_gradients(zip(bert_gradients, self.bert_variables))
        normal_gradients = gradients[self.num_bert:]
        self.optimizer.apply_gradients(zip(normal_gradients, self.normal_variables))
