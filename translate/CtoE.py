import matplotlib as mpl
import matplotlib.pyplot as plt
#matplotlib inline
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras

en_fra_file_path = 'D:\\Python\\new.txt'

#文本标准化
import unicodedata
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c) !='Mn') # 如果一个unicode是由多个ASCII组成的，就将其拆开;如果不是重音的话
en_sentence = 'Cheers!'
fr_sentence = 'À votre santé !'
print(unicode_to_ascii(en_sentence))
print(unicode_to_ascii(fr_sentence)) # 忽略了e的重音符号
 
 
import re
def preprocess_sentence(s):
    '''将标点符号和正文区分开'''
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([¿,?.!])",r" \1 ",s) # 标点符号前后加空格
    s = re.sub(r"[' ']+",r" ",s) # 空格去重
    s = re.sub(r"[^a-zA-Z¿,?.!]",r" ",s)
    s = s.rstrip().strip() # 去掉后前空格
    s = '<start> '+s + ' <end>' # 添加起止符
    return s
 
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(fr_sentence))

 
def parse_data(filename):
    lines = open(filename,encoding="UTF-8").read().strip().split('\n')
    sentence_pairs = [line.split('\t') for line in lines]
    preprocessed_sentence_pairs =[
        (preprocess_sentence(en),preprocess_sentence(cm)) for en,cm in sentence_pairs
    ]
    
    return zip(*preprocessed_sentence_pairs) # 返回英文列表和法语列表
 
en_dataset,fr_dataset = parse_data(en_fra_file_path)
print(en_dataset[-1])
print(fr_dataset[-1])

 
def tokenizer(lang):
    '''文本id化'''
    lang_tokenizer = keras.preprocessing.text.Tokenizer(
        num_words=None,#默认None处理所有字词
        filters='',#过滤特殊字符
        split=' '#字符串，单词的分隔符
    )
    lang_tokenizer.fit_on_texts(lang) # 构建词表
    tensor = lang_tokenizer.texts_to_sequences(lang) # 转化操作
    tensor = keras.preprocessing.sequence.pad_sequences(tensor,padding = 'post') # 句子后做padding，将序列转化为经过填充以后的一个长度相同的新序列新序列
    return tensor,lang_tokenizer
 
input_tensor,input_tokenizer = tokenizer(fr_dataset[0:30000])
output_tensor,output_tokenizer = tokenizer(en_dataset[0:30000]) # trick 为了加快运行速度限制了数据集的大小
 
def max_length(tensor):
    return max(len(t) for t in tensor)
 
max_length_input = max_length(input_tensor)
max_length_output = max_length(output_tensor)
print(max_length_input)
print(max_length_output)
 
 
from sklearn.model_selection import train_test_split # 训练和测试集切分
input_train,input_eval,output_train,output_eval =  train_test_split(
    input_tensor,
    output_tensor,
    test_size = 0.2
)
print(len(input_train),len(input_eval),len(output_train),len(output_eval))
 
 
def convert(example,tokenizer):
    '''查看tokenizer是否正常工作'''
    for t in example:
        if t !=0:
            print('%d --> %s'%(t,tokenizer.index_word[t])) # 查看词表中的词
            
convert(input_train[0],input_tokenizer)
convert(output_train[0],output_tokenizer)
 
 
def make_dataset(input_tensor,output_tensor,batch_size,epochs,shuffle):
    '''转化为高效的数据管道，加载数据集'''#建立数据管道，要训​​练该模型，需要一个数据管道来为其提供标记的训练数据
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor,output_tensor)
    )#接收tensor,对tensor第一维度进行切分，numpy创建tf
    if shuffle: #打乱数据
        dataset = dataset.shuffle(30000)
    dataset = dataset.repeat(epochs).batch(batch_size,drop_remainder=True) # 不足batch就丢掉， 先repeat再batch是对于重复后数组的分组，保证次序
    return dataset

batch_size = 64 #一次喂入64个数据,一次训练选取的样本数
epochs = 20 #整个数据集训练次数

train_dataset = make_dataset(input_train,output_train,batch_size,epochs,True) 
test_dataset = make_dataset(input_eval,output_eval,batch_size,1,False)

for x,y in train_dataset.take(1): 
    print("x,y",x.shape,y.shape)
    print("x",x)
    print("y",y)

embedding_units = 256
units = 1024 # rnn的输出维度
input_vocab_size = len(input_tokenizer.word_index)+1
output_vocab_size = len(output_tokenizer.word_index)+1
 
 
# 设置GPU内存自增
class Encoder(keras.Model):
    def __init__(self,vocab_size,embedding_units,encoding_units,batch_size):
        # encoding units是lstm的大小
        super(Encoder,self).__init__()
        self.batch_size = batch_size
        self.encoding_units = encoding_units
        self.embedding = keras.layers.Embedding(vocab_size,embedding_units)#对每个词编号后将编号变成词向量，将正整数转换为固定尺寸的稠密向量
        self.gru = keras.layers.GRU(self.encoding_units,return_sequences=True,return_state=True,
                                   recurrent_initializer='glorot_uniform') # lstm的变种，遗忘门  = 1-输入门
        
    
    def call(self,x,hidden):
        x = self.embedding(x)
        # rnn自适应长度
        output,state = self.gru(x,initial_state=hidden) # 得到每一步的输出和最后一步的隐含状态
        return output,state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size,self.encoding_units))
    
encoder = Encoder(input_vocab_size,embedding_units,units,batch_size)
sample_hidden = encoder.initialize_hidden_state() # 初始化隐藏层参数
sample_output,sample_hidden = encoder(x,sample_hidden)
print("sample_hidden shape ",sample_hidden.shape) # (64,16,1024),1024就是encoder_units
print("sample_output shape ",sample_output.shape) # 64,1024
 
 
 
class BahdanauAttention(keras.Model):
    # 实现Attention机制
    def __init__(self,units):
        super(BahdanauAttention,self).__init__()
        self.W1 = keras.layers.Dense(units) # 为decoder和encoder分别做一个全连接
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)
        
    def call(self,decoder_hidden,encoder_outputs): # decode某一步的隐含状态，encoder的每一步输出
        # decoder_hidden.shape = (batch_size,units)
        # encoder_outputs.shape = (batch_size,length,units)
        # 为了实现加和，需要对decoder进行扩展
        decoder_hidden_with_time_axis = tf.expand_dims(decoder_hidden,axis=1)
        # before V: (batch_size,length,units)
        # after V: (batch_size,length,1)
        score = self.V(
                    tf.nn.tanh(
                        self.W1(encoder_outputs) + self.W2(decoder_hidden_with_time_axis)))
        # shape:(batch_size,length,1)
        attention_weights = tf.nn.softmax(score,axis=1)
        # context_vector.shape = (batch_size,length,units)
        context_vector = attention_weights*encoder_outputs # broadcast
        # (batchsize,units)
        context_vector = tf.reduce_sum(context_vector,axis=1)
        return context_vector,attention_weights
    
attention_model = BahdanauAttention(units=10)
attention_results,attention_weights = attention_model(sample_hidden,sample_output)
print("attention reuslts shape ",attention_results.shape)
print("attention weights shape ",attention_weights.shape)
 
 
class Decoder(keras.Model):
    def __init__(self,vocab_size,embedding_units,decoding_units,batch_size):
        super(Decoder,self).__init__()
        self.batch_size = batch_size
        self.decoding_units = decoding_units
        self.embedding = keras.layers.Embedding(vocab_size,embedding_units)
        self.gru = keras.layers.GRU(self.decoding_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.fc = keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.decoding_units)
    
    def call(self,x,hidden,encoding_outputs): # hidden是上一步的输出
        # context_v:(batch_size,units)
        context_vector,attention_weights = self.attention(hidden,encoding_outputs)
        # before embedding :x (batchsize,1)
        # afeter embedding: x (batchsize,1,embedding_units)
        x = self.embedding(x)
        # 最后一维度大小相加
        combined_x = tf.concat([
            tf.expand_dims(context_vector,axis=1),x],axis=-1) #https://blog.csdn.net/leviopku/article/details/82380118 相当于在最后一维进行拼接
        # output.shape:(batchsize,1,decoding_units)
        # state.shape:batchsize,decoding_units
        output,state = self.gru(combined_x)
        # outputshape:(batchsize,decoding_units)生成二维会舍去维度为1的维度
        output = tf.reshape(output,(-1,output.shape[2]))
        # out.shape:(batchsize,vocab_size)
        output = self.fc(output)
        return output,state,attention_weights
decoder = Decoder(output_vocab_size,embedding_units,units,batch_size)
outputs = decoder(tf.random.uniform((batch_size,1)),
                 sample_hidden,sample_output)
decoder_output,decoder_hidden,decoder_aw = outputs
print("decoder_output shape ",decoder_output.shape)
print("decoder_hidden shape", decoder_hidden.shape)
print("decoder_aw shape",decoder_aw.shape)


optimizer = keras.optimizers.Adam()
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                        reduction='none') # 结果的输出没有被激活函数，所以from_logits=True
def loss_function(real,pred):
    '''计算单步损失函数'''
    mask = tf.math.logical_not(tf.math.equal(real,0)) # 取反操作，排除paddding的损失函数
    loss_ = loss_object(real,pred)
    mask = tf.cast(mask,dtype=loss_.dtype)
    loss_*=mask # 防止padding部分污染操作
    return tf.reduce_mean(loss_)
 
 
@tf.function
def train_step(inp,targ,encoding_hidden):
    '''多步计算损失函数'''
    loss = 0
    with tf.GradientTape() as tape:
        encoding_outputs,encoding_hidden = encoder(inp,encoding_hidden) # 进行模型计算
        decoding_hidden = encoding_hidden
        '''
        e.g. <start> I am here <end>
        1. <start> -> I
        2. I -> am {同时具有<start>的信息}
        3. am -> here
        4. here -> end
        '''
        for t in range(0,targ.shape[1]-1):
            decoding_input = tf.expand_dims(targ[:,t],1) # 实际输入的应该是一个已有的targ序列，而不仅仅是单步的targ值
            predictions,decoding_hidden,_ = decoder(
            decoding_input,decoder_hidden,encoding_outputs
            )
            loss += loss_function(targ[:,t+1],predictions)
    batch_loss = loss / int(targ.shape[0])
    variables = encoder.trainable_variables+decoder.trainable_variables
    gradients = tape.gradient(loss,variables) # batch loss也可以
    optimizer.apply_gradients(zip(gradients,variables))
    return batch_loss
 
 
 
epochs = 10
steps_per_epoch = len(input_tensor)//batch_size

# 模型训练
for epoch in range(epochs):
    start = time.time()
    encoding_hidden =  encoder.initialize_hidden_state()
    total_loss = 0
    for (batch,(inp,targ)) in enumerate(train_dataset.take(steps_per_epoch)):
        batch_loss  = train_step(inp,targ,encoding_hidden)
        total_loss +=batch_loss
        
        
        if batch %100==0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")   # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data, float)




def evalute(input_sentence):
    attention_matrix = np.zeros((max_length_output,max_length_input)) # 可以理解为每个输出值都和输入值有一定的关系
    input_sentence = preprocess_sentence(input_sentence) # 文本预处理
    inputs = [input_tokenizer.word_index[token] for token in input_sentence.split(' ')] # 文本id化
    inputs =keras.preprocessing.sequence.pad_sequences(
        [inputs],maxlen=max_length_input,padding='post'
    ) # padding
    inputs =tf.convert_to_tensor(inputs)
    result = ''
    # encoding_hidden = encoder.initialize_hidden_state()
    encoding_hidden = tf.zeros((1,units))
    encoding_outputs,encoding_hidden = encoder(inputs,encoding_hidden)
    decoding_hidden = encoding_hidden
    # e.g. <start> ->A
    # A ->B ->C->D
    # decoding_input.shape [1,1]
    # 这里注意与train的不同，train是直接使用targ,evaluate是使用上一步输出
    decoding_input = tf.expand_dims([output_tokenizer.word_index['<start>']],0)
    for t in range(max_length_output):
        predictions,decoding_hidden,attention_weights = decoder(decoding_input,decoding_hidden,encoding_outputs)
        # attentionweights.shape(batchsize,inputlength,1) ->(1,16,1)
        attention_weights = tf.reshape(attention_weights,(-1,)) # shape为16的向量
        attention_matrix[t] = attention_weights.numpy()
        # predictions.shape :(batchszie,vocabsize) (1,4935)
        predicted_id =tf.argmax(predictions[0]).numpy() # 取最大概率的id作为下一步预测
        result += output_tokenizer.index_word[predicted_id] + ' '
        if output_tokenizer.index_word[predicted_id] =='<end>':
            return result,input_sentence,attention_matrix
        decoding_input = tf.expand_dims([predicted_id],0)
    return result,input_sentence,attention_matrix
        
def plot_attention(attention_matrix,input_sentence,predicted_sentence):
    fig  = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.matshow(attention_matrix,cmap='viridis') # 配色方案
    font_dict = {'fontsize':14}
    ax.set_xticklabels(['']+input_sentence,fontdict=font_dict,rotation=90)
    ax.set_yticklabels(['']+ predicted_sentence,fontdict=font_dict,)
    plt.show()
    
def translate(input_sentence):
    '''综合调用'''
    results,input_sentence,attention_matrix =evalute(input_sentence)
    print("Input: %s"%(input_sentence))
    print("Predicted translation:%s"%(results))
    
    # 有些reuslt不一定能达到max_length_output的长度，padding部分也要去掉
    attention_matrix = attention_matrix[:len(results.split(' ')),:len(input_sentence.split(' '))]
    plot_attention(attention_matrix,input_sentence.split(' '),results.split(' '))
 
 
translate(u'Ils sont très grands.') # 不能出现没有的词，不然会报keyerror,就是在index索引的时候出现问题



