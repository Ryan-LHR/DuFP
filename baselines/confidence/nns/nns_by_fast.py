import numpy as np

class NNS:
    """
    NNS (implementation of FAST, modified for torch framework)
    """
    def __init__(self, model, layer_index, classes):
        self.classes = classes
        self.layer_index = layer_index
        self.model = model
        self.hidden_model = keras.Model(inputs=self.model.inputs, outputs=self.model.layers[layer_index].output)
        self.follow_model = tf.keras.Sequential(model.layers[layer_index + 1:])
        self.follow_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fetch_internals(self, x):
        x_hidden_layer_values = []
        batch_size = 1000
        for i in range(x.shape[0] // batch_size):
            x_batch = x_[i * batch_size:(i + 1) * batch_size]
            hidden_values_batch = self.hidden_model(x_batch)
            item = hidden_values_batch
            if item.ndim == 4:
                item = item.numpy().reshape(item.shape[0], -1, item.shape[-1])
                item = np.average(item, axis=1)
            x_hidden_layer_values.append(item)

        hidden_layer_values = np.concatenate(x_hidden_layer_values)
        self.internals = hidden_layer_values
        self.probs = self.model.predict(x)

    def cosine_similarity(self, vector, matrix):
        dot_product = np.dot(matrix, vector)
        vector_norm = np.linalg.norm(vector)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        cosine_sim = dot_product / (vector_norm * matrix_norms)
        return cosine_sim

    def euclidean_distance(self, vector, matrix):
        diff = matrix - vector
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        return dist

    def smooth(self, x, k, a):
        org = self.hidden_model(x)
        org = org.numpy().reshape(org.shape[0], -1, org.shape[-1])
        org = np.average(org, axis=1)[0]
        cos_dis = 1 - self.cosine_similarity(org, self.internals)
        index = np.argsort(cos_dis)[:k]
        probs = self.probs[index]
        x_prob = self.model.predict(x)[0]
        final_prob = a * x_prob + (1 - a) * (1 / k) * (np.sum(probs, axis=0))
        return final_prob


class NNS_NORM:
    def __init__(self, model, layer_index, classes):
        self.classes = classes
        self.layer_index = layer_index
        self.class_freqs = []
        self.class_ns = []
        self.class_patterns = []
        self.model = model
        self.hidden_model = keras.Model(inputs=self.model.inputs, outputs=self.model.layers[layer_index].output)
        self.follow_model = tf.keras.Sequential(model.layers[layer_index + 1:])
        self.follow_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fetch_internals(self, x, y):
        x_hidden_layer_values = []
        batch_size = 1000
        for i in range(x.shape[0] // batch_size):
            x_batch = x[i * batch_size:(i + 1) * batch_size]
            hidden_values_batch = self.hidden_model(x_batch)
            item = hidden_values_batch
            if item.ndim == 4:
                item = item.numpy().reshape(item.shape[0], -1, item.shape[-1])
                item = np.average(item, axis=1)
            x_hidden_layer_values.append(item)

        matrix = np.concatenate(x_hidden_layer_values)
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized_matrix = matrix / matrix_norms

        self.internals = normalized_matrix
        self.probs = self.model.predict(x)

    def cosine_similarity(self, vector, matrix):
        dot_product = np.dot(matrix, vector)
        vector_norm = np.linalg.norm(vector)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        cosine_sim = dot_product / (vector_norm * matrix_norms)
        return cosine_sim

    def euclidean_distance(self, vector, matrix):
        diff = matrix - vector
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        return dist

    def smooth(self, x, k, a):
        org = self.hidden_model(x)
        org = org.numpy().reshape(org.shape[0], -1, org.shape[-1])
        org = np.average(org, axis=1)[0]
        org_norm = np.linalg.norm(org)
        org = org / org_norm
        cos_dis = 1 - self.cosine_similarity(org, self.internals)
        index = np.argsort(cos_dis)[:k]
        probs = self.probs[index]
        x_prob = self.model.predict(x)[0]
        final_prob = a * x_prob + (1 - a) * (1 / k) * (np.sum(probs, axis=0))

        return final_prob