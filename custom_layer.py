# -*- coding: utf-8 -*-
import numpy as np
from keras import backend as K
from keras.layers import Layer

class AttentionRNN(Layer):

    def __init__(self, units, dropout=0., **kwargs):
        self.units = units  # 输出维度
        self.dropout = min(1., max(0., dropout))
        super(AttentionRNN, self).__init__(**kwargs)

    def build(self, input_shape):  # 定义可训练参数
        self.embedding_kernel = self.add_weight(name='embedding_kernel',
                                                shape=(input_shape[-1], self.units),
                                                initializer='glorot_normal',
                                                trainable=True)
        self.state_kernel = self.add_weight(name='state_kernel',
                                            shape=(self.units, self.units),
                                            initializer='glorot_normal',
                                            trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer='glorot_normal',
                                    trainable=True)
        self.length_kernel = self.add_weight(name='length_kernel',
                                             shape=(self.units, 1),
                                             initializer='glorot_normal',
                                             trainable=True)
        self.rnn_state_kernel = self.add_weight(name='rnn_state_kernel',
                                                shape=(self.units, self.units * 3),
                                                initializer='glorot_normal',
                                                trainable=True)

        self.rnn_embedding_kernel = self.add_weight(name='rnn_embedding_kernel',
                                                    shape=(input_shape[-1], self.units * 3),
                                                    initializer='glorot_normal',
                                                    trainable=True)

        self.rnn_newin_kernel = self.add_weight(name='rnn_embedding_kernel',
                                                shape=(input_shape[-1], self.units * 3),
                                                initializer='glorot_normal',
                                                trainable=True)
        '''
        self.rnn_bias = self.add_weight(name='rnn_bias',
                                      shape=(self.units*3,),
                                      initializer='glorot_normal',
                                      trainable=True)
        '''
        self.rnn_state_u = self.rnn_state_kernel[:, :self.units]
        self.rnn_state_r = self.rnn_state_kernel[:, self.units: self.units * 2]
        self.rnn_state_h = self.rnn_state_kernel[:, self.units * 2:]

        self.rnn_embedding_u = self.rnn_embedding_kernel[:, :self.units]
        self.rnn_embedding_r = self.rnn_embedding_kernel[:, self.units: self.units * 2]
        self.rnn_embedding_h = self.rnn_embedding_kernel[:, self.units * 2:]

        self.rnn_newin_u = self.rnn_newin_kernel[:, :self.units]
        self.rnn_newin_r = self.rnn_newin_kernel[:, self.units: self.units * 2]
        self.rnn_newin_h = self.rnn_newin_kernel[:, self.units * 2:]
        '''
        self.rnn_bias_u = self.rnn_bias[:self.units]
        self.rnn_bias_r = self.rnn_bias[self.units: self.units * 2]
        self.rnn_bias_h = self.rnn_bias[self.units * 2:]
'''

    def step_do(self, step_in, states):  # 定义每一步的迭代

        if 0 < self.dropout < 1.:
            self._dropout_mask = K.in_train_phase(K.dropout(K.ones_like(step_in), self.dropout), K.ones_like(step_in))
            self.r_dropout_mask = K.in_train_phase(K.dropout(K.ones_like(states[0]), self.dropout),
                                                   K.ones_like(states[0]))
        print('states[0]', states[0].shape)
        state = states[0]
        if 0 < self.dropout < 1.:
            in_value = step_in * self._dropout_mask
            state = state * self.r_dropout_mask

        print(step_in.shape)
        seq_length = K.shape(states)[0]
        query = K.dot(state, self.state_kernel)
        query = K.expand_dims(query, axis=-2)
        print('query:', query.shape)
        first = K.expand_dims(states[1], axis=-2)
        second = K.expand_dims(states[2], axis=-2)
        third = K.expand_dims(in_value, axis=-2)
        value = K.concatenate([first, second, third], axis=-2)
        embedding = K.reshape(value, shape=(-1, self.units))
        embedding = K.dot(embedding, self.embedding_kernel)
        print('embedding:', embedding.shape)
        embedding = K.reshape(embedding, shape=(-1, seq_length, self.units))
        attention_prob = K.tanh(embedding + query + self.bias)
        attention_prob = K.reshape(attention_prob, shape=(-1, self.units))
        attention_prob = K.dot(attention_prob, self.length_kernel)
        attention_prob = K.reshape(attention_prob, shape=(-1, seq_length))
        attention_prob_max = K.max(attention_prob, axis=1, keepdims=True)
        attention_prob = K.softmax(attention_prob - attention_prob_max)
        # attention_prob = K.softmax(attention_prob)
        attention_prob = K.expand_dims(attention_prob, axis=2)
        # attention_prob = K.tanh(K.dot(states[0], self.state_kernel) + K.dot(step_in, self.embedding_kernel))
        # attention_prob = K.softmax(attention_prob)
        attention_out = K.sum(value * attention_prob, axis=-2)  # [batch_size x units]

        u_p = K.sigmoid(K.dot(state, self.rnn_state_u) + K.dot(attention_out, self.rnn_embedding_u) + K.dot(in_value,
                                                                                                            self.rnn_newin_u))  # +self.rnn_bias_u)
        r_p = K.sigmoid(K.dot(state, self.rnn_state_r) + K.dot(attention_out, self.rnn_embedding_r) + K.dot(in_value,
                                                                                                            self.rnn_newin_r))  # +self.rnn_bias_r)
        out_candidate = K.tanh(
            r_p * K.dot(state, self.rnn_state_h) + K.dot(attention_out, self.rnn_embedding_h) + K.dot(in_value,
                                                                                                      self.rnn_newin_h))  # +self.rnn_bias_h)
        step_out = (1 - u_p) * state + u_p * out_candidate
        '''
        h_state = K.softmax(K.dot(states[0],self.rnn_state_u) + K.dot(attention_out,self.rnn_state_r))
        step_out = K.softmax(K.dot(h_state,self.rnn_state_h))
        '''
        print('step_out', step_out.shape)
        return step_out, [step_out, states[2], step_in]

    def call(self, inputs, training=None):  # 定义正式执行的函数
        init_states = [K.zeros((K.shape(inputs)[0], self.units)), K.zeros((K.shape(inputs)[0], K.shape(inputs)[-1])),
                       K.zeros((K.shape(inputs)[0], K.shape(inputs)[-1]))]  # 定义初始态(全零)
        print('inputs', K.shape(inputs)[0])
        outputs = K.rnn(self.step_do, inputs, init_states, unroll=False)  # 循环执行step_do函数
        # print('outputs',outputs[0].shape)
        return outputs[0]
        # return successive_outputs # outputs是一个tuple，outputs[0]为最后时刻的输出，
        # outputs[1]为整个输出的时间序列，output[2]是一个list，
        # 是中间的隐藏状态。

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


class SelfAttention(Layer):

    def __init__(self, units, **kwargs):
        self.units = units  # 输出维度
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):  # 定义可训练参数
        self.query_kernel = self.add_weight(name='query_kernel',
                                            shape=(input_shape[-1], self.units),
                                            initializer='glorot_normal',
                                            trainable=True)

        self.key_kernel = self.add_weight(name='key_kernel',
                                          shape=(input_shape[-1], self.units),
                                          initializer='glorot_normal',
                                          trainable=True)

        self.value_kernel = self.add_weight(name='value_kernel',
                                            shape=(input_shape[-1], self.units),
                                            initializer='glorot_normal',
                                            trainable=True)

        print('input_shape', input_shape)

    def call(self, inputs):  # 定义正式执行的函数
        print(self.query_kernel.shape)
        print('inputs', inputs.shape)
        query = K.dot(inputs, self.query_kernel)
        print('query', query.shape)

        key = K.permute_dimensions(K.dot(inputs, self.key_kernel), (0, 2, 1))
        print('key', key.shape)

        value = K.dot(inputs, self.value_kernel)
        print('value', value.shape)

        attention_prob = K.batch_dot(key, query, axes=[1, 2]) / np.sqrt(self.units)
        attention_prob = K.softmax(attention_prob)
        print(attention_prob.shape)
        print(inputs.shape)
        outputs = K.batch_dot(attention_prob, value)
        print(outputs.shape)

        # outputs = K.mean(outputs, axis=-2, keepdims=False)
        # outputs = K.sum(inputs * attention_prob, axis=-2)

        print(outputs[:, -1].shape)
        # print(outputs.shape)
        return outputs[:, -1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


class Self_RNN(Layer):

    def __init__(self, units, dropout=0., **kwargs):
        self.units = units # 输出维度
        self.dropout = min(1., max(0., dropout))
        super(Self_RNN, self).__init__(**kwargs)
    
    def build(self, input_shape): # 定义可训练参数
        self.query_kernel = self.add_weight(name='query_kernel',
                                      shape=(input_shape[-1], self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.key_kernel = self.add_weight(name='key_kernel',
                                      shape=(input_shape[-1], self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.value_kernel = self.add_weight(name='value_kernel',
                                      shape=(input_shape[-1], self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.rec_kernel = self.add_weight(name='rec_kernel',
                                      shape=(self.units, self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        print('input_shape',input_shape)

    def step_do(self, step_in, states): # 定义每一步的迭代
        
        in_value = step_in
        if 0 < self.dropout < 1.:
            self._dropout_mask = K.in_train_phase(K.dropout(K.ones_like(step_in), self.dropout), K.ones_like(step_in))
        if 0 < self.dropout < 1.:
            in_value = step_in * self._dropout_mask
        
        hist=K.dot(states[0], self.rec_kernel)
        
        first = K.expand_dims(hist,axis=-2)
        second = K.expand_dims(states[1],axis=-2)
        third = K.expand_dims(in_value,axis=-2)
        inp = K.concatenate([first,second,third], axis=-2)
        
        print(self.query_kernel.shape)
        print('inp',inp.shape)
        query=K.dot(inp, self.query_kernel)
        print('query', query.shape)
        
        key=K.permute_dimensions(K.dot(inp, self.key_kernel),(0,2,1))
        print('key', key.shape)
        
        value=K.dot(inp, self.value_kernel)
        print('value', value.shape)
        
        attention_prob=K.batch_dot(key, query, axes=[1, 2])/np.sqrt(self.units)
        attention_prob = K.softmax(attention_prob)
        print(attention_prob.shape)
        outputs = K.batch_dot(attention_prob, value)
        print(outputs.shape)
        return outputs[:,0], [outputs[:,0], step_in]
    
    def call(self, inputs): # 定义正式执行的函数
        
        init_states = [K.zeros((K.shape(inputs)[0],self.units)), K.zeros((K.shape(inputs)[0],K.shape(inputs)[-1]))] # 定义初始态(全零)
        print('inputs',K.shape(inputs)[0])
        outputs = K.rnn(self.step_do, inputs, init_states, unroll=False) # 循环执行step_do函数
        #print('outputs',outputs[0].shape)
        return outputs[0]
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class Bi_Self_RNN(Layer):

    def __init__(self, units, dropout=0., **kwargs):
        self.units = units  # 输出维度
        self.dropout = min(1., max(0., dropout))
        self.supports_masking = True
        super(Bi_Self_RNN, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        output_mask = None
        return output_mask

    def build(self, input_shape):  # 定义可训练参数
        print(input_shape[-1])
        self.query_kernel = self.add_weight(name='query_kernel',
                                            shape=(input_shape[-1], self.units),
                                            initializer='glorot_normal',
                                            trainable=True)

        self.key_kernel = self.add_weight(name='key_kernel',
                                          shape=(input_shape[-1], self.units),
                                          initializer='glorot_normal',
                                          trainable=True)

        self.value_kernel = self.add_weight(name='value_kernel',
                                            shape=(input_shape[-1], self.units),
                                            initializer='glorot_normal',
                                            trainable=True)

        '''
        self.rec_kernel = self.add_weight(name='rec_kernel',
                                      shape=(self.units, self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        '''
        print('input_shape', input_shape)

    def step_do(self, step_in, states):  # 定义每一步的迭代

        in_value = step_in
        if 0 < self.dropout < 1.:
            self._dropout_mask = K.in_train_phase(K.dropout(K.ones_like(step_in), self.dropout), K.ones_like(step_in))
        if 0 < self.dropout < 1.:
            in_value = step_in * self._dropout_mask

        # hist = K.tanh(K.dot(states[0], self.rec_kernel))
        # hist = K.tanh(states[0])

        in_value = K.expand_dims(in_value, axis=-2)

        l_state = K.expand_dims(states[0], axis=-2)
        l_inp = K.concatenate([l_state, in_value], axis=-2)

        s_state = K.expand_dims(states[1], axis=-2)
        s_inp = K.concatenate([s_state, in_value], axis=-2)

        l_query = K.dot(l_inp, self.query_kernel)

        l_key = K.dot(l_inp, self.key_kernel)

        l_value = K.dot(l_inp, self.value_kernel)

        l_attention_prob = K.batch_dot(l_query, l_key, axes=[2, 2]) / np.sqrt(self.units)
        print(l_attention_prob.shape)
        l_attention_prob = K.softmax(l_attention_prob)
        l_outputs = K.batch_dot(l_attention_prob, l_value)
        l_outputs = K.tanh(l_outputs)

        s_query = K.dot(s_inp, self.query_kernel)

        s_key = K.dot(s_inp, self.key_kernel)

        s_value = K.dot(s_inp, self.value_kernel)

        s_attention_prob = K.batch_dot(s_query, s_key, axes=[2, 2]) / np.sqrt(self.units)
        s_attention_prob = K.softmax(s_attention_prob)
        s_outputs = K.batch_dot(s_attention_prob, s_value)
        s_outputs = K.tanh(s_outputs)

        lt = K.expand_dims(l_outputs[:, 0], axis=-2)
        st = K.expand_dims(s_outputs[:, 1], axis=-2)
        outputs = K.concatenate([lt, st], axis=-2)

        query = K.dot(outputs, self.query_kernel)

        key = K.dot(outputs, self.key_kernel)

        value = K.dot(outputs, self.value_kernel)

        attention_prob = K.batch_dot(query, key, axes=[2, 2]) / np.sqrt(self.units)
        attention_prob = K.softmax(attention_prob)
        print(attention_prob.shape)
        att_out = K.batch_dot(attention_prob, value, axes=[2, 1])

        # outputs = K.concatenate([l_outputs[:,0], s_outputs[:,1]], axis=-1)
        # outputs = 0.5*l_outputs[:,0] + 0.5*s_outputs[:,1]
        print('inner_outputs.shape', outputs.shape)
        return att_out[:, 0], [att_out[:, 0], att_out[:, 1]]
        # return outputs, [l_outputs[:,0], s_outputs[:,1]]

    def call(self, inputs):  # 定义正式执行的函数

        # init_states = [K.zeros((K.shape(inputs)[0],self.units)), K.zeros((K.shape(inputs)[0],self.units))] # 定义初始态(全零)
        init_states = [inputs[:, 0], inputs[:, 0]]
        # print('inputs',K.shape(inputs)[0])
        print('inputs', inputs.shape)
        outputs = K.rnn(self.step_do, inputs[:, 1:], init_states, unroll=False)  # 循环执行step_do函数
        # print('outputs',outputs[0].shape)
        return outputs[0]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class similar_RNN(Layer):

    def __init__(self, units, dropout=0., **kwargs):
        self.units = units # 输出维度
        self.dropout = min(1., max(0., dropout))
        self.supports_masking = True
        super(similar_RNN, self).__init__(**kwargs)
    
    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        output_mask =  None
        return output_mask
    
    def build(self, input_shape): # 定义可训练参数
        
        self.query_kernel = self.add_weight(name='query_kernel',
                                      shape=(input_shape[-1]*2, self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.key_kernel = self.add_weight(name='key_kernel',
                                      shape=(input_shape[-1]*2, self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.value_kernel = self.add_weight(name='value_kernel',
                                      shape=(input_shape[-1]*2, self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        #print('input_shape',input_shape)

    def step_do(self, step_in, states): # 定义每一步的迭代
        
        in_value = step_in
        if 0 < self.dropout < 1.:
            self._dropout_mask = K.in_train_phase(K.dropout(K.ones_like(step_in), self.dropout), K.ones_like(step_in))
        if 0 < self.dropout < 1.:
            in_value = step_in * self._dropout_mask
        
        d1 = K.sigmoid(K.sqrt(K.sum((K.square(in_value-states[0])/self.units),axis=-1,keepdims=True)/2))
        d2 = K.sigmoid(K.sqrt(K.sum((K.square(in_value-states[1])/self.units),axis=-1,keepdims=True)/2))
        '''
        d1 = K.sigmoid(K.sum((K.abs(in_value-states[0])/self.units),axis=-1,keepdims=True))
        d2 = K.sigmoid(K.sum((K.abs(in_value-states[1])/self.units),axis=-1,keepdims=True))
        '''
        print('d1.shape',d1.shape)
        state1 = d1*states[0] + (1-d1)*in_value
        print('state1.shape',state1.shape)
        state2 = (1-d2)*states[0] + d2*in_value
        '''
        lt = K.expand_dims(state1,axis=-2)
        st = K.expand_dims(state2,axis=-2)
        outputs = K.concatenate([lt, st], axis=-2)
        '''
        outputs = K.concatenate([state1, state2], axis=-1)
        
        return outputs, [state1, state2]
    
    def call(self, inputs): # 定义正式执行的函数
        
        init_states = [K.zeros((K.shape(inputs)[0],K.shape(inputs)[-1])), K.zeros((K.shape(inputs)[0],K.shape(inputs)[-1]))] # 定义初始态(全零)
        #init_states = [inputs[:,0], inputs[:,0]]
        #print('inputs',K.shape(inputs)[0])
        outputs = K.rnn(self.step_do, inputs, init_states, unroll=False) # 循环执行step_do函数
        #print('outputs[1]',outputs.shape)
        
        print('outputs[0].shape',outputs[0].shape)
        
        query=K.dot(outputs[1], self.query_kernel)
        print('query.shape',query.shape)
        key=K.dot(outputs[1], self.key_kernel)
        
        value=K.dot(outputs[1], self.value_kernel)
        
        attention_prob = K.batch_dot(query, key, axes=[2, 2])/np.sqrt(self.units)
        attention_prob = K.softmax(attention_prob)
        print(attention_prob.shape)
        att_out = K.batch_dot(attention_prob, value, axes=[2, 1])
        
        return att_out[:,-1]
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class similar_RNN_multi(Layer):

    def __init__(self, units, dropout=0., **kwargs):
        self.units = units # 输出维度
        self.dropout = min(1., max(0., dropout))
        self.supports_masking = True
        super(similar_RNN_multi, self).__init__(**kwargs)
    
    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        output_mask =  None
        return output_mask
    
    def build(self, input_shape): # 定义可训练参数
        
        self.query_kernel = self.add_weight(name='query_kernel',
                                      shape=(input_shape[-1]*2, self.units*4),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.key_kernel = self.add_weight(name='key_kernel',
                                      shape=(input_shape[-1]*2, self.units*4),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.value_kernel = self.add_weight(name='value_kernel',
                                      shape=(input_shape[-1]*2, self.units*4),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.switch_kernel = self.add_weight(name='switch_kernel',
                                      shape=(self.units*4, self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        #print('input_shape',input_shape)
        
        self.query_kernel1 = self.query_kernel[:, :self.units]
        self.query_kernel2 = self.query_kernel[:, self.units:self.units*2]
        self.query_kernel3 = self.query_kernel[:, self.units*2:self.units*3]
        self.query_kernel4 = self.query_kernel[:, self.units*3:]
        
        self.key_kernel1 = self.key_kernel[:, :self.units]
        self.key_kernel2 = self.key_kernel[:, self.units:self.units*2]
        self.key_kernel3 = self.key_kernel[:, self.units*2:self.units*3]
        self.key_kernel4 = self.key_kernel[:, self.units*3:]
        
        self.value_kernel1 = self.value_kernel[:, :self.units]
        self.value_kernel2 = self.value_kernel[:, self.units:self.units*2]
        self.value_kernel3 = self.value_kernel[:, self.units*2:self.units*3]
        self.value_kernel4 = self.value_kernel[:, self.units*3:]
        
        

    def step_do(self, step_in, states): # 定义每一步的迭代
        
        in_value = step_in
        if 0 < self.dropout < 1.:
            self._dropout_mask = K.in_train_phase(K.dropout(K.ones_like(step_in), self.dropout), K.ones_like(step_in))
        if 0 < self.dropout < 1.:
            in_value = step_in * self._dropout_mask
        '''
        d1 = K.sigmoid(K.sqrt(K.sum((K.square(in_value-states[0])/self.units),axis=-1,keepdims=True)))
        d2 = K.sigmoid(K.sqrt(K.sum((K.square(in_value-states[1])/self.units),axis=-1,keepdims=True)))
        
        
        d1 = K.sigmoid(K.sum((K.abs(in_value-states[0])/self.units),axis=-1,keepdims=True))
        d2 = K.sigmoid(K.sum((K.abs(in_value-states[1])/self.units),axis=-1,keepdims=True))
        '''
        d1 = 0.85
        d2 = 0.85
        
        #print('d1.shape',d1.shape)
        state1 = d1*states[0] + (1-d1)*in_value
        print('state1.shape',state1.shape)
        state2 = (1-d2)*states[0] + d2*in_value
        '''
        lt = K.expand_dims(state1,axis=-2)
        st = K.expand_dims(state2,axis=-2)
        outputs = K.concatenate([lt, st], axis=-2)
        '''
        outputs = K.concatenate([state1, state2], axis=-1)
        
        return outputs, [state1, state2]
    
    def call(self, inputs): # 定义正式执行的函数
        
        init_states = [K.zeros((K.shape(inputs)[0],K.shape(inputs)[-1])), K.zeros((K.shape(inputs)[0],K.shape(inputs)[-1]))] # 定义初始态(全零)
        #init_states = [inputs[:,0], inputs[:,0]]
        #print('inputs',K.shape(inputs)[0])
        outputs = K.rnn(self.step_do, inputs, init_states, unroll=False) # 循环执行step_do函数
        #print('outputs[1]',outputs.shape)
        
        print('outputs[0].shape',outputs[0].shape)
        
        query1=K.dot(outputs[1], self.query_kernel1)
        
        key1=K.dot(outputs[1], self.key_kernel1)
        
        value1=K.dot(outputs[1], self.value_kernel1)
        
        attention_prob1 = K.batch_dot(query1, key1, axes=[2, 2])/np.sqrt(self.units)
        attention_prob1 = K.softmax(attention_prob1)
        att_out1 = K.batch_dot(attention_prob1, value1, axes=[2, 1])
        
        query2=K.dot(outputs[1], self.query_kernel2)
        
        key2=K.dot(outputs[1], self.key_kernel2)
        
        value2=K.dot(outputs[1], self.value_kernel2)
        
        attention_prob2 = K.batch_dot(query2, key2, axes=[2, 2])/np.sqrt(self.units)
        attention_prob2 = K.softmax(attention_prob2)
        att_out2 = K.batch_dot(attention_prob2, value2, axes=[2, 1])
        
        query3=K.dot(outputs[1], self.query_kernel3)
        
        key3=K.dot(outputs[1], self.key_kernel3)
        
        value3=K.dot(outputs[1], self.value_kernel3)
        
        attention_prob3 = K.batch_dot(query3, key3, axes=[2, 2])/np.sqrt(self.units)
        attention_prob3 = K.softmax(attention_prob3)
        att_out3 = K.batch_dot(attention_prob3, value3, axes=[2, 1])
        
        query4=K.dot(outputs[1], self.query_kernel4)
        
        key4=K.dot(outputs[1], self.key_kernel4)
        
        value4=K.dot(outputs[1], self.value_kernel4)
        
        attention_prob4 = K.batch_dot(query4, key4, axes=[2, 2])/np.sqrt(self.units)
        attention_prob4 = K.softmax(attention_prob4)
        att_out4 = K.batch_dot(attention_prob4, value4, axes=[2, 1])
        
        att_out = K.concatenate([att_out1, att_out2, att_out3, att_out4], axis=-1)
        out = K.dot(att_out, self.switch_kernel)
        return out[:, -1]
      
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class genres_similar(Layer):

    def __init__(self, units, dropout=0., **kwargs):
        self.units = units # 输出维度
        self.dropout = min(1., max(0., dropout))
        self.supports_masking = True
        super(genres_similar, self).__init__(**kwargs)
    
    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        output_mask =  None
        return output_mask
    
    def build(self, input_shape): # 定义可训练参数
        '''
        self.state_kernel = self.add_weight(name='state_kernel',
                                      shape=(input_shape[-1], self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.input_kernel = self.add_weight(name='input_kernel',
                                      shape=(input_shape[-1], self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        '''
        self.query_kernel = self.add_weight(name='query_kernel',
                                      shape=(self.units*2, self.units*4),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.key_kernel = self.add_weight(name='key_kernel',
                                      shape=(self.units*2, self.units*4),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.value_kernel = self.add_weight(name='value_kernel',
                                      shape=(self.units*2, self.units*4),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.switch_kernel = self.add_weight(name='switch_kernel',
                                      shape=(self.units*4, self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        #print('input_shape',input_shape)
        
        self.query_kernel1 = self.query_kernel[:, :self.units]
        self.query_kernel2 = self.query_kernel[:, self.units:self.units*2]
        self.query_kernel3 = self.query_kernel[:, self.units*2:self.units*3]
        self.query_kernel4 = self.query_kernel[:, self.units*3:]
        
        self.key_kernel1 = self.key_kernel[:, :self.units]
        self.key_kernel2 = self.key_kernel[:, self.units:self.units*2]
        self.key_kernel3 = self.key_kernel[:, self.units*2:self.units*3]
        self.key_kernel4 = self.key_kernel[:, self.units*3:]
        
        self.value_kernel1 = self.value_kernel[:, :self.units]
        self.value_kernel2 = self.value_kernel[:, self.units:self.units*2]
        self.value_kernel3 = self.value_kernel[:, self.units*2:self.units*3]
        self.value_kernel4 = self.value_kernel[:, self.units*3:]
        

    def step_do(self, step_in, states): # 定义每一步的迭代
        
        in_value = step_in
        if 0 < self.dropout < 1.:
            self._dropout_mask = K.in_train_phase(K.dropout(K.ones_like(step_in), self.dropout), K.ones_like(step_in))
        if 0 < self.dropout < 1.:
            in_value = step_in * self._dropout_mask
        '''    
        g1 = states[0][:, :-18]
        g2 = states[1][:, :-18]
        g3 = in_value[:, :-18]
        
        d1 = K.sigmoid(K.sqrt(K.sum((K.square(g3-g1)),axis=-1,keepdims=True)))
        d2 = K.sigmoid(K.sqrt(K.sum((K.square(g3-g2)),axis=-1,keepdims=True)))
        
        d1 = K.sigmoid(K.sum((K.abs(in_value-states[0])/self.units),axis=-1,keepdims=True))
        d2 = K.sigmoid(K.sum((K.abs(in_value-states[1])/self.units),axis=-1,keepdims=True))
        '''
        
        d1 = K.sigmoid(states[0]*in_value)/2
        d2 = K.sigmoid(states[1]*in_value)/2
        #update=1#K.sigmoid(K.dot(states[0], self.state_kernel)+ K.dot(step_in, self.input_kernel))
        ''''''
        #print('d1.shape',d1.shape)
        state1 = d1*states[0] + (1-d1)*in_value
        print('state1.shape',state1.shape)
        state2 = (1-d2)*states[1] + d2*in_value
        
        #outputs = (1-update)*states[0]+update*step_in
        
        
        '''
        lt = K.expand_dims(state1,axis=-2)
        st = K.expand_dims(state2,axis=-2)
        outputs = K.concatenate([lt, st], axis=-2)
        
        out1 = K.dot(state1, self.encode_kernel)
        out2 = K.dot(state2, self.encode_kernel)
        '''
        outputs = K.concatenate([state1, state2], axis=-1)
        #outputs = K.relu(outputs)
        
        return outputs, [state1, state2]
    
    def call(self, inputs): # 定义正式执行的函数
        
        init_states = [K.zeros((K.shape(inputs)[0],K.shape(inputs)[-1])), K.zeros((K.shape(inputs)[0],K.shape(inputs)[-1]))] # 定义初始态(全零)
        #init_states = [inputs[:,0], inputs[:,0]]
        #print('inputs',K.shape(inputs)[0])
        outputs = K.rnn(self.step_do, inputs, init_states, unroll=False) # 循环执行step_do函数
        #print('outputs[1]',outputs.shape)
        
        print('outputs[0].shape',outputs[0].shape)
        
        query1=K.dot(outputs[1], self.query_kernel1)
        
        key1=K.dot(outputs[1], self.key_kernel1)
        
        value1=K.dot(outputs[1], self.value_kernel1)
        
        attention_prob1 = K.batch_dot(query1, key1, axes=[2, 2])/np.sqrt(self.units)
        attention_prob1 = K.softmax(attention_prob1)
        att_out1 = K.batch_dot(attention_prob1, value1, axes=[2, 1])
        
        query2=K.dot(outputs[1], self.query_kernel2)
        
        key2=K.dot(outputs[1], self.key_kernel2)
        
        value2=K.dot(outputs[1], self.value_kernel2)
        
        attention_prob2 = K.batch_dot(query2, key2, axes=[2, 2])/np.sqrt(self.units)
        attention_prob2 = K.softmax(attention_prob2)
        att_out2 = K.batch_dot(attention_prob2, value2, axes=[2, 1])
        
        query3=K.dot(outputs[1], self.query_kernel3)
        
        key3=K.dot(outputs[1], self.key_kernel3)
        
        value3=K.dot(outputs[1], self.value_kernel3)
        
        attention_prob3 = K.batch_dot(query3, key3, axes=[2, 2])/np.sqrt(self.units)
        attention_prob3 = K.softmax(attention_prob3)
        att_out3 = K.batch_dot(attention_prob3, value3, axes=[2, 1])
        
        query4=K.dot(outputs[1], self.query_kernel4)
        
        key4=K.dot(outputs[1], self.key_kernel4)
        
        value4=K.dot(outputs[1], self.value_kernel4)
        
        attention_prob4 = K.batch_dot(query4, key4, axes=[2, 2])/np.sqrt(self.units)
        attention_prob4 = K.softmax(attention_prob4)
        att_out4 = K.batch_dot(attention_prob4, value4, axes=[2, 1])
        
        att_out = K.concatenate([att_out1, att_out2, att_out3, att_out4], axis=-1)
        out = K.dot(att_out, self.switch_kernel)
        return out[:, -1]
      
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
class weight_RNN_multi(Layer):

    def __init__(self, units, dropout=0., **kwargs):
        self.units = units # 输出维度
        self.dropout = min(1., max(0., dropout))
        self.supports_masking = True
        super(weight_RNN_multi, self).__init__(**kwargs)
    
    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        output_mask =  None
        return output_mask
    
    def build(self, input_shape): # 定义可训练参数
        
        self.state_kernel = self.add_weight(name='state_kernel',
                                      shape=(input_shape[-1], self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.input_kernel = self.add_weight(name='input_kernel',
                                      shape=(input_shape[-1], self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.query_kernel = self.add_weight(name='query_kernel',
                                      shape=(input_shape[-1], self.units*4),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.key_kernel = self.add_weight(name='key_kernel',
                                      shape=(input_shape[-1], self.units*4),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.value_kernel = self.add_weight(name='value_kernel',
                                      shape=(input_shape[-1], self.units*4),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.switch_kernel = self.add_weight(name='switch_kernel',
                                      shape=(self.units*4, self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
        #print('input_shape',input_shape)
        
        self.query_kernel1 = self.query_kernel[:, :self.units]
        self.query_kernel2 = self.query_kernel[:, self.units:self.units*2]
        self.query_kernel3 = self.query_kernel[:, self.units*2:self.units*3]
        self.query_kernel4 = self.query_kernel[:, self.units*3:]
        
        self.key_kernel1 = self.key_kernel[:, :self.units]
        self.key_kernel2 = self.key_kernel[:, self.units:self.units*2]
        self.key_kernel3 = self.key_kernel[:, self.units*2:self.units*3]
        self.key_kernel4 = self.key_kernel[:, self.units*3:]
        
        self.value_kernel1 = self.value_kernel[:, :self.units]
        self.value_kernel2 = self.value_kernel[:, self.units:self.units*2]
        self.value_kernel3 = self.value_kernel[:, self.units*2:self.units*3]
        self.value_kernel4 = self.value_kernel[:, self.units*3:]
        
        

    def step_do(self, step_in, states): # 定义每一步的迭代
        
        in_value = step_in
        if 0 < self.dropout < 1.:
            self._dropout_mask = K.in_train_phase(K.dropout(K.ones_like(step_in), self.dropout), K.ones_like(step_in))
        if 0 < self.dropout < 1.:
            in_value = step_in * self._dropout_mask
        '''
        d1 = K.sigmoid(K.sqrt(K.sum((K.square(in_value-states[0])/self.units),axis=-1,keepdims=True)))
        d2 = K.sigmoid(K.sqrt(K.sum((K.square(in_value-states[1])/self.units),axis=-1,keepdims=True)))
        
        d1 = K.sigmoid(K.sum((K.abs(in_value-states[0])/self.units),axis=-1,keepdims=True))
        d2 = K.sigmoid(K.sum((K.abs(in_value-states[1])/self.units),axis=-1,keepdims=True))
        '''
        #print('d1.shape',d1.shape)
        update = K.sigmoid(K.dot(states[0],self.state_kernel)+K.dot(in_value,self.input_kernel))
        outputs = (1-update)*states[0]+update*in_value
        '''
        lt = K.expand_dims(state1,axis=-2)
        st = K.expand_dims(state2,axis=-2)
        outputs = K.concatenate([lt, st], axis=-2)
        '''
        #outputs = K.concatenate([state1, state2], axis=-1)
        
        return outputs, [outputs]
    
    def call(self, inputs): # 定义正式执行的函数
        
        init_states = [K.zeros((K.shape(inputs)[0],K.shape(inputs)[-1]))] # 定义初始态(全零)
        #init_states = [inputs[:,0], inputs[:,0]]
        #print('inputs',K.shape(inputs)[0])
        outputs = K.rnn(self.step_do, inputs, init_states, unroll=False) # 循环执行step_do函数
        #print('outputs[1]',outputs.shape)
        
        print('outputs[0].shape',outputs[0].shape)
        
        query1=K.dot(outputs[1], self.query_kernel1)
        
        key1=K.dot(outputs[1], self.key_kernel1)
        
        value1=K.dot(outputs[1], self.value_kernel1)
        
        attention_prob1 = K.batch_dot(query1, key1, axes=[2, 2])/np.sqrt(self.units)
        attention_prob1 = K.softmax(attention_prob1)
        att_out1 = K.batch_dot(attention_prob1, value1, axes=[2, 1])
        
        query2=K.dot(outputs[1], self.query_kernel2)
        
        key2=K.dot(outputs[1], self.key_kernel2)
        
        value2=K.dot(outputs[1], self.value_kernel2)
        
        attention_prob2 = K.batch_dot(query2, key2, axes=[2, 2])/np.sqrt(self.units)
        attention_prob2 = K.softmax(attention_prob2)
        att_out2 = K.batch_dot(attention_prob2, value2, axes=[2, 1])
        
        query3=K.dot(outputs[1], self.query_kernel3)
        
        key3=K.dot(outputs[1], self.key_kernel3)
        
        value3=K.dot(outputs[1], self.value_kernel3)
        
        attention_prob3 = K.batch_dot(query3, key3, axes=[2, 2])/np.sqrt(self.units)
        attention_prob3 = K.softmax(attention_prob3)
        att_out3 = K.batch_dot(attention_prob3, value3, axes=[2, 1])
        
        query4=K.dot(outputs[1], self.query_kernel4)
        
        key4=K.dot(outputs[1], self.key_kernel4)
        
        value4=K.dot(outputs[1], self.value_kernel4)
        
        attention_prob4 = K.batch_dot(query4, key4, axes=[2, 2])/np.sqrt(self.units)
        attention_prob4 = K.softmax(attention_prob4)
        att_out4 = K.batch_dot(attention_prob4, value4, axes=[2, 1])
        
        att_out = K.concatenate([att_out1, att_out2, att_out3, att_out4], axis=-1)
        out = K.dot(att_out, self.switch_kernel)
        return out[:, -1]
      
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
