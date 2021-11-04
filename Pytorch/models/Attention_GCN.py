import tensorflow as tf

def __attn_coeffs(data, adj_mat, _C, name='coefficient'):
    with tf.variable_scope(name) as scope:
        X1 = tf.transpose(data, [0,2,1])
        X2 = tf.einsum('ij,ajk->aik', _C, X1)
        attn_matrix = tf.matmul(data, X2)
        attn_matrix = tf.multiply(adj_mat, attn_matrix)
        attn_matrix = tf.nn.tanh(attn_matrix)

        return attn_matrix

def attn(data, adj_mat, output_dim, num_head, do_skip_connect=False, name='Attention'):
    with tf.variable_scope(name) as scope:
        states = []
        for k in range(num_head):
            C_k = tf.get_variable('attn_weight-'+str(k), 
                                  initializer=tf.contrib.layers.xavier_initializer(), 
                                  shape=[output_dim, output_dim], dtype=tf.float64)
            X_k = tf.layers.dense(data, output_dim, use_bias=True)
            attn_matrix = __attn_coeffs(X_k, adj_mat, C_k)
            X_k = tf.matmul(attn_matrix, X_k)
            states.append(X_k)

        output = tf.reduce_mean(states, 0)
        if do_skip_connect:
            output = tf.add(data, output)
        output = tf.nn.relu(output)

        return output
    
class create():
    def __init__(self, num_nodes, num_features, num_output, num_head, do_skip_connect=False,
                 gpu_memory_fraction=None, optimizer_type='adam', phase='train'):
        
        if phase not in ['train', 'inference'] : 
            raise  ValueError("phase must be 'train' or 'inference'.")
            
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_output = num_output
        self.num_head = num_head
        self.do_skip_connect = do_skip_connect
        
        self.graph = tf.get_default_graph()
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        self.sess = tf.Session(config=config, graph=self.graph) 
        
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float64, shape=(None, self.num_nodes, self.num_features))
            self.A = tf.placeholder(tf.float64, shape=(None, self.num_nodes, self.num_nodes))
            self.valid = tf.placeholder(tf.float64, shape=(None, self.num_nodes))
            self.Y_truth = tf.placeholder(tf.float64, shape=(None, self.num_output))
            
            self.__create_model()
            
            if phase=='train':
                self.lr = tf.placeholder(tf.float32, name="lr")
                self.loss = tf.losses.mean_squared_error(self.x, self.output)
                
                self.sess.run(tf.global_variables_initializer())
                self.train_op = self.__set_op(self.loss, self.lr, optimizer_type)

                uninit_vars = [v for v in tf.global_variables()
                              if not tf.is_variable_initialized(v).eval(session=self.sess)]
                self.sess.run(tf.variables_initializer(uninit_vars))
    
    def __create_model(self):
        for idx, chn in enumerate([128, 128, 256, 256, 512, 512, 1024, 1024]):
            output = self.X if idx == 0 else output
            output = attn(output, self.A, chn, self.num_head, self.do_skip_connect, f'Attn_{idx+1}')
            
        self.output = []
        for i in range(self.num_nodes):
            self.output.append(tf.layers.dense(output[:,i], units=1, use_bias=True))
        self.output = tf.concat(self.output, axis=-1)
        self.output *= valid
        
    def __set_op(self, loss_op, learning_rate, optimizer_type="adam"):
        with self.graph.as_default():
            if optimizer_type=="adam":
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif optimizer_type == "adagrad":
                optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.0001)
            elif optimizer_type == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            elif optimizer_type == "momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
            elif optimizer_type == "adadelta":
                optimizer = tf.train.AdadeltaOptimizer(learning_rate,rho=0.95,epsilon=1e-09)
            else : raise ValueError("{} optimizer doesn't exist.".format(optimizer_type))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss_op)

        return train_op