import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(8)
import matplotlib.pyplot as pltf
import numpy as np

folder_path = '/mnt/6a35bb5c-3c75-400e-a2fa-146fee20967e/home/fcssaleh/ICWT/BCH_127_64'
Parity_Check_Matrix = np.load(folder_path+'/test_data/H_BCH_127_64.npy')
rows, columns = np.shape(Parity_Check_Matrix)
gragh_edge = np.sum(Parity_Check_Matrix)
batch_size = 40

###################################### construction of weights #####################################################
hidden_1_layer_c = np.zeros([columns,gragh_edge])
edge_number = np.zeros([rows,columns])
k = 0
for i in range(rows):
    for j in range(columns):
       if Parity_Check_Matrix[i][j] == 1:
           hidden_1_layer_c[:,k] = Parity_Check_Matrix[i,:]
           hidden_1_layer_c[j,k] = 0
           k +=1
           edge_number[i][j] = k 
h = np.matrix.transpose(hidden_1_layer_c)
w = np.array([[h[j] for _ in range(40)] for j in range(gragh_edge)] )
_w = np.ones(np.shape(w))-w
c1 = tf.convert_to_tensor(w,dtype='float32')

hidden_2_layer_v = np.zeros([gragh_edge,gragh_edge])
k = 0
for i in range(rows):
    for j in range(columns):
       if Parity_Check_Matrix[i][j] == 1:
          for t in range(rows):
              if edge_number[t][j] != 0:
                  index = np.int(edge_number[t][j])
                  hidden_2_layer_v[index-1,k] = 1
          index = np.int(edge_number[i][j])
          hidden_2_layer_v[index-1,k] = 0 
          k +=1
v1 = tf.convert_to_tensor(hidden_2_layer_v,dtype='float32')

hidden_3_layer_c = np.zeros([gragh_edge,gragh_edge])
k = 0
for i in range(rows):
    for j in range(columns):
       if Parity_Check_Matrix[i][j] == 1:
          for t in range(columns):
              if edge_number[i][t] != 0:
                  index = np.int(edge_number[i][t])
                  hidden_3_layer_c[index-1,k] = 1
          index = np.int(edge_number[i][j])
          hidden_3_layer_c[index-1,k] = 0 
          k +=1   
h = np.matrix.transpose(hidden_3_layer_c)
w = np.array([[h[j] for _ in range(40)] for j in range(gragh_edge)] )
_w2 = np.ones(np.shape(w))-w
c2 = tf.convert_to_tensor(w,dtype='float32')

output_v = np.zeros([gragh_edge,columns])
for i in range(rows):
    for j in range(columns):
       if Parity_Check_Matrix[i][j] == 1:
          for t in range(rows):
              if edge_number[t][j] != 0:
                  index = np.int(edge_number[t][j])
                  output_v[index-1,j] = 1
v2 = tf.convert_to_tensor(output_v,dtype='float32')

biase_sc = np.zeros([columns,gragh_edge],dtype=np.float32)
for k in range(batch_size):
  for i in range(rows):
    for j in range(columns):
      if edge_number[i][j] != 0:
        index = np.int(edge_number[i][j])
        biase_sc[j,index-1] = 1.0 

##################################  Functions  ####################################

folder_path = '/mnt/6a35bb5c-3c75-400e-a2fa-146fee20967e/home/fcssaleh/ICWT/BCH_127_64'
code_generatorMatrix = np.load(folder_path+'/test_data/G_BCH_127_64.npy')
x_v = np.load(folder_path+'/test_data/small_x_v.npy')
y_v = np.load(folder_path+'/test_data/small_y_v.npy')
code_n = 127
code_k = 64

code_rate = 1.0*code_k/code_n

# init the AWGN
start_snr = 1
stop_snr = 8
word_seed = 786000
noise_seed = 345000   

batch_size = 40
numOfWordSim_train = 5
batches_for_val = np.array([10,10,10,30,40,100,300,600])
batch_in_epoch = 1000
num_of_batch = 10000000
train_on_zero_word = True
test_on_zero_word = False
snr_db = np.arange(start_snr,stop_snr+1,1,dtype=np.float32)
snr_lin = 10.0**(snr_db/10.0)
scaling_factor = np.sqrt(1.0/(2.0*snr_lin*code_rate)) 
wordRandom = np.random.RandomState(word_seed)
random = np.random.RandomState(noise_seed)
    
def create_mix_epoch(scaling_factor, wordRandom, numOfWordSim, code_n, code_k, code_generatorMatrix, is_zeros_word):

    X = np.zeros([1,code_n], dtype=np.float32)
    Y = np.zeros([1,code_n], dtype=np.int64)

    # build set for epoch
    for sf_i in scaling_factor:
        if is_zeros_word:
            infoWord_i = 0*wordRandom.randint(0, 2, size=(numOfWordSim, code_k))
        else:
            infoWord_i = wordRandom.randint(0, 2, size=(numOfWordSim, code_k))

        Y_i = np.dot(infoWord_i, code_generatorMatrix) % 2
        X_p_i = random.normal(0.0,1.0,Y_i.shape)*sf_i + (-1)**(1-Y_i)
        x_llr_i = 2*X_p_i/(sf_i**2)
        X = np.vstack((X,x_llr_i))
        Y = np.vstack((Y,Y_i))

    X = X[1:]
    Y = Y[1:]

    return X,Y

def calc_ber_fer(snr_db, Y_v_pred, Y_v, numOfWordSim):
    ber_test = np.zeros(snr_db.shape[0])
    fer_test = np.zeros(snr_db.shape[0])
    Y_v_pred_i = Y_v_pred[0:numOfWordSim[0],:]
    Y_v_i = Y_v[0:numOfWordSim[0],:]
    ber_test[0] = np.abs(((Y_v_pred_i>0.5)-Y_v_i)).sum()/(Y_v_i.shape[0]*Y_v_i.shape[1])
    fer_test[0] = (np.abs(np.abs(((Y_v_pred_i>0.5)-Y_v_i))).sum(axis=1)>0).sum()*1.0/Y_v_i.shape[0]
    for i in range(1,snr_db.shape[0]):
        Y_v_pred_i = Y_v_pred[np.sum(numOfWordSim[0:i]):np.sum(numOfWordSim[0:i+1]),:]
        Y_v_i = Y_v[np.sum(numOfWordSim[0:i]):np.sum(numOfWordSim[0:i+1]),:]
        ber_test[i] = np.abs(((Y_v_pred_i>0.5)-Y_v_i)).sum()/(Y_v_i.shape[0]*Y_v_i.shape[1])

    return ber_test#, fer_test

############################################################## construction  of network ##########################################################
x = tf.placeholder('float',[None,columns],name='x')
y = tf.placeholder('float')
batch_size = 40
clip_tanh = 10.0
threshold = 0.99999997

l1 = tf.tanh(0.5 * x)
    
l2 = tf.multiply(c1,l1)
l2 = l2 + _w
l2 = tf.reduce_prod(l2,axis=2)
l2 = tf.transpose(l2)
l2  = tf.clip_by_value(l2,-threshold ,threshold )
l2 = 2 * tf.atanh(l2)
    
output0 = tf.matmul(l2, v2) + x
    
hidden_3_layer = {'weights':tf.Variable(v1),'biase':tf.Variable(biase_sc)}
hidden_3_layer['weights'] = tf.multiply(v1,hidden_3_layer['weights'])
hidden_3_layer['biase'] = tf.multiply(biase_sc,hidden_3_layer['biase']) 
l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']),tf.matmul(x,hidden_3_layer['biase']))
l3 = tf.tanh(0.5 * l3)
    
hidden_5_layer = {'weights':tf.Variable(v1),'biase':tf.Variable(biase_sc)}
hidden_5_layer['weights'] = tf.multiply(v1,hidden_5_layer['weights'])
hidden_5_layer['biase'] = tf.multiply(biase_sc,hidden_5_layer['biase'])
l5 = tf.add(tf.matmul(l3, hidden_5_layer['weights']),tf.matmul(x,hidden_5_layer['biase']))
l5 = tf.tanh(0.5 * l5)
    
l4 = tf.multiply(c2,l5)
l4 = l4 + _w2
l4 = tf.reduce_prod(l4,axis=2)
l4 = tf.transpose(l4)
l4  = tf.clip_by_value(l4,-threshold ,threshold)
l4 = 2 * tf.atanh(l4)
    
output1 = tf.matmul(l4, v2) + x
    
hidden_7_layer = {'weights':tf.Variable(v1),'biase':tf.Variable(biase_sc)}
hidden_7_layer['weights'] = tf.multiply(v1,hidden_7_layer['weights'])
hidden_7_layer['biase'] = tf.multiply(biase_sc,hidden_7_layer['biase'])
l7 = tf.add(tf.matmul(l4, hidden_7_layer['weights']),tf.matmul(x,hidden_7_layer['biase']))
l7 = tf.tanh(0.5 * l7)
    
hidden_9_layer = {'weights':tf.Variable(v1),'biase':tf.Variable(biase_sc)}
hidden_9_layer['weights'] = tf.multiply(v1,hidden_9_layer['weights'])
hidden_9_layer['biase'] = tf.multiply(biase_sc,hidden_9_layer['biase'])
l9 = tf.add(tf.matmul(l7, hidden_9_layer['weights']),tf.matmul(x,hidden_9_layer['biase']))
l9 = tf.tanh(0.5 * l9)
    
l6 = tf.multiply(c2,l9)
l6 = l6 + _w2
l6 = tf.reduce_prod(l6,axis=2)
l6 = tf.transpose(l6)
l6  = tf.clip_by_value(l6,-threshold ,threshold )
l6 = 2 * tf.atanh(l6)
output2 = tf.matmul(l6, v2) + x    

hidden_11_layer = {'weights':tf.Variable(v1),'biase':tf.Variable(biase_sc)}
hidden_11_layer['weights'] = tf.multiply(v1,hidden_11_layer['weights'])
hidden_11_layer['biase'] = tf.multiply(biase_sc,hidden_11_layer['biase'])
l11 = tf.add(tf.matmul(l6, hidden_11_layer['weights']),tf.matmul(x,hidden_11_layer['biase']))
l11 = tf.tanh(0.5 * l11)
    
hidden_13_layer = {'weights':tf.Variable(v1),'biase':tf.Variable(biase_sc)}
hidden_13_layer['weights'] = tf.multiply(v1,hidden_13_layer['weights'])
hidden_13_layer['biase'] = tf.multiply(biase_sc,hidden_13_layer['biase'])
l13 = tf.add(tf.matmul(l11, hidden_13_layer['weights']),tf.matmul(x,hidden_13_layer['biase']))
l13 = tf.tanh(0.5 * l13)
    
l8 = tf.multiply(c2,l13)
l8 = l8 + _w2
l8 = tf.reduce_prod(l8,axis=2)
l8 = tf.transpose(l8)
l8  = tf.clip_by_value(l8,-threshold ,threshold )
l8 = 2 * tf.atanh(l8)
        
marginal_1_layer = {'weights':tf.Variable(v2)} 
marginal_1_layer['weights'] = tf.multiply(v2,marginal_1_layer['weights'])
output3 = tf.matmul(l8, marginal_1_layer['weights']) + x 
sigout = tf.nn.sigmoid(output3)
       
################################################# construction  of network for training state #############################################

cost0 = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=output0,labels=y))
cost1 = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=output1,labels=y))
cost2 = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=output2,labels=y))
cost3 = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=output3,labels=y))

cost = cost0+ cost1+ cost2+ cost3
train_step = tf.train.RMSPropOptimizer(0.003).minimize(cost)

train_loss_vec = np.zeros(1,dtype=np.float32)
BER = np.zeros([1,8],dtype=np.float32)
val_loss_vec = np.zeros(1,dtype=np.float32)

with tf.Session() as sess:
    
    sess.run(tf.initialize_all_variables())
    
    for i in range(num_of_batch):                
        training_data, training_labels = create_mix_epoch(scaling_factor, wordRandom, numOfWordSim_train, code_n, code_k, code_generatorMatrix, is_zeros_word=train_on_zero_word)
        training_labels_for_mse = training_labels
        y_train, train_loss, _ = sess.run(fetches=[output3, cost, train_step], feed_dict={x: training_data, y: training_labels_for_mse})
       
        if(i%batch_in_epoch == 0):


            print('Finish Epoch - ', i/batch_in_epoch)
            y_v_pred = np.zeros([1,code_n], dtype=np.float32)
            loss_v = np.zeros([1, 1], dtype=np.float32)
            n_snr = -1
            vector_number = 0
            for k_sf in scaling_factor:
                n_snr += 1
                for j in range(batches_for_val[n_snr]):
                    start = batch_size*j + vector_number
                    stop = batch_size*(j+1) + vector_number
                    x_v_j = x_v[start:stop,:]
                    y_v_j = y_v[start:stop,:]
                    y_v_pred_j, loss_v_j = sess.run(fetches = [sigout, cost] ,feed_dict={x:x_v_j, y:y_v_j})

                    y_v_pred = np.vstack((y_v_pred,y_v_pred_j))
                    loss_v = np.vstack((loss_v, loss_v_j))
                vector_number = vector_number+batches_for_val[n_snr]*batch_size

            ber_val = calc_ber_fer(snr_db, y_v_pred[1:,:], y_v, batch_size*batches_for_val)

            print('SNR[dB] validation - ', snr_db)
            print('BER validation - ', ber_val)
            
            train_loss_vec = np.vstack((train_loss_vec, train_loss))
            val_loss_vec = np.vstack((val_loss_vec, np.mean(loss_v)))
            BER = np.vstack((BER, ber_val))
            np.save('N2_127_64.npy',BER[1:,:])
            pltf.figure()
            pltf.plot(train_loss_vec[1:])
            pltf.plot(val_loss_vec[1:], color='red')
            pltf.xlabel('epoch');
            pltf.ylabel('loss')
            pltf.legend(('Train', 'Validation'))
            pltf.grid(True)
            pltf.savefig('N2_127_64.png')
            pltf.close()
            pltf.figure()
            pltf.plot(np.sum(BER[1:,:],axis=1))
            pltf.xlabel('epoch');
            pltf.ylabel('sum_of_BER')
            pltf.grid(True)
            pltf.savefig('N2_127_64_sum_of_BER.png')
            pltf.close()
            
            
    
