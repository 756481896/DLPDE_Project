#         with tf.variable_scope('forward',reuse=True) as scope:
#             x0 = tf.constant(value=-math.pi,shape=[1,1])
#             input_x0 = tf.concat([t,x0],1)
#             u_x0 = self.neural_net(input_x0,scope)

#         with tf.variable_scope('forward',reuse=True) as scope:
#             x1 = tf.constant(value=math.pi,shape=[1,1])
#             input_x1 = tf.concat([t,x1],1)
#             u_x1 = self.neural_net(input_x1,scope)

#         with tf.variable_scope('forward',reuse=True) as scope:
#             t0 = tf.constant(value=0.0,shape=[1,1])
#             input_t0 = tf.concat([t0,x],1)
#             u_t0 = self.neural_net(input_t0,scope)