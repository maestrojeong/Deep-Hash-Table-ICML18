#./
from dist import masked_minimum, masked_maximum

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, logging_ops, math_ops, nn
import tensorflow as tf

def pairwise_distance_w_obj1(feature, objective):
    '''
    Defined on docs/idea.pdf
    Args:
        feature - 2-D tensor of size [number of data, feature dimensions]
        objective - 2-D tensor of size [number of data, feature dimensions]
            should be binary 0 or 1
    Return:
        pairwise : 2-D Tensor of size [number of data, number of data] 
    '''
    mask = 1.0 - objective # active 0 non active 1 
    mask = tf.multiply(tf.expand_dims(mask, axis=1), tf.expand_dims(mask, axis=0)) # [number of data, number of data, feature dimensions] both non active 1 otherwise 0 
    mask = 1.0 - mask # both non active 0 otherwise 1  

    pairwise = tf.reduce_sum(
                    tf.multiply( 
                        tf.abs(
                            tf.subtract(
                                tf.expand_dims(feature, axis=1),\
                                tf.expand_dims(feature, axis=0))
                            ),
                        mask
                        ),
                    axis=-1) # [number of data, number of data] 
    # To make distance positive
    return pairwise

def pairwise_similarity_w_obj1(anc, pos, obj):
    '''
    multiple
    Args:
        anc - 2D tensor [ndata, embedding_dim]
        pos - 2D tensor [ndata, embedding_dim]
        obj - 2D tensor [ndata, embedding_dim]
            which is binary
    Return :
        simlarity - 2D tensor [ndata, ndata] 
    '''
    # mask[i][j] = 1 if (i,j) not active  else 0
    mask = 1.0 - obj # [ndata, embedding_dim]
    # mask[i][j][k] = 1 if (i,k) and (j,k) not active  else 0
    mask = tf.multiply(tf.expand_dims(mask, axis=1), tf.expand_dims(mask, axis=0)) # [ndata, ndata, embedding_dim]
    # mask[i][j][k] = 0 if (i,k) and (j,k) not active  else 1
    mask = 1.0 - mask # [ndata, ndata, embedding_dim]

    similarity = tf.reduce_sum(
                    tf.multiply(
                        tf.multiply(
                            tf.expand_dims(anc, axis=1),
                            tf.expand_dims(pos, axis=0)),
                        mask),
                    axis=-1) # [ndata, ndata]
    return similarity

def pairwise_similarity_w_obj2(anc, pos, obj):
    '''
    -subtract l1 norm
    Args:
        anc - 2D tensor [ndata, embedding_dim]
        pos - 2D tensor [ndata, embedding_dim]
        obj - 2D tensor [ndata, embedding_dim]
            which is binary
    Return :
        simlarity - 2D tensor [ndata, ndata] 
    '''
    # mask[i][j] = 1 if (i,j) not active  else 0
    mask = 1.0 - obj # [ndata, embedding_dim]
    # mask[i][j][k] = 1 if (i,k) and (j,k) not active  else 0
    mask = tf.multiply(tf.expand_dims(mask, axis=1), tf.expand_dims(mask, axis=0)) # [ndata, ndata, embedding_dim]
    # mask[i][j][k] = 0 if (i,k) and (j,k) not active  else 1
    mask = 1.0 - mask # [ndata, ndata, embedding_dim]

    # Get per pair similarities.
    similarity = tf.negative(tf.reduce_sum(
                                tf.multiply( 
                                    tf.abs(tf.subtract(tf.expand_dims(anc, axis=1), tf.expand_dims(pos, axis=0))),
                                    mask),
                                axis=-1)) # [ndata, ndata]
    return similarity

def triplet_semihard_loss_hash(labels, embeddings, objectives, pairwise_distance, margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.
    Args:
	labels - 1-D tensor [batch_size] as tf.int32
                multiclass integer labels.
	embeddings - 2-D tensor [batch_size, feature dimensions] as tf.float32
                     `Tensor` of embedding vectors. Embeddings should be l2 normalized.
        pairwise_distance - func 
            args : 
                feature - 2D tensor [ndata, feature dims], 
                objective - 2D tensor [ndata, feature_dims]
            return :
                2D tensor [ndata, ndata] 
	margin - float defaults to be 1.0
                margin term in the loss definition.

    Returns:
    	triplet_loss - tf.float32 scalar.
    """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # pdist_matrix[i][j] = dist(i, j)
    pdist_matrix = pairwise_distance(embeddings, objectives) # [batch_size, batch_size]
    # adjacency[i][j]=1 if label[i]==label[j] else 0
    adjacency = math_ops.equal(labels, array_ops.transpose(labels)) # [batch_size, batch_size]
    # adjacency_not[i][j]=0 if label[i]==label[j] else 0
    adjacency_not = math_ops.logical_not(adjacency) # [batch_size, batch_size]
    batch_size = array_ops.size(labels)

    # Compute the mask.
    # pdist_matrix_tile[batch_size*i+j, k] = distance(j, k)
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1]) # [batch_size*batch_size, batch_size]
    # reshape(transpose(pdist_matrix), [-1, 1])[batch_size*i+j][0] = distance(j, i)
    # tile(adjacency_not, [batch_size, 1])[batch_size*i+j][k] = 1 if label[j]!=label[k] otherwise 0
    # mask[batch_size*i+j][k] = 1 if label[j]!=label[k] different label, and distance(j,k)>distance(j,i)
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(pdist_matrix_tile, array_ops.reshape(array_ops.transpose(pdist_matrix), [-1, 1]))) # [batch_size*batch_size, batch_size]
    # mask_final[i][j]=1 if there exists k s.t. label[j]!=label[k] and distance(j,k)>distance(j,i)
    mask_final = array_ops.reshape(
        math_ops.greater(math_ops.reduce_sum(math_ops.cast(mask, dtype=dtypes.float32), 1, keep_dims=True), 0.0),
        [batch_size, batch_size])# [batch_size, batch_size]
    # mask_final[i][j]=1 if there exists k s.t. label[i]!=label[k] and distance(i,k)>distance(i,j)
    mask_final = array_ops.transpose(mask_final)# [batch_size, batch_size]

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32) # [batch_size, batch_size]
    mask = math_ops.cast(mask, dtype=dtypes.float32) # [batch_size*batch_size, batch_size]

    # masked_minimum(pdist_matrix_tile, mask)[batch*i+j][1] = pdist_matrix[j][k] s.t minimum over 'k's label[j]!=label[k], distance(j,k)>distance(j,i)
    # negatives_outside[i][j] = pdist_matrix[j][k] s.t minimum over 'k's label[j]!=label[k], distance(j,k)>distance(j,i)
    negatives_outside = array_ops.reshape(masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]) # [batch_size, batch_size]
    # negatives_outside[i][j] = pdist_matrix[i][k] s.t minimum over 'k's label[i]!=label[k], distance(i,k)>distance(i,j)
    negatives_outside = array_ops.transpose(negatives_outside) # [batch_size, batch_size]

    # negatives_inside[i][j] = pdist_matrix[i][k] s.t maximum over label[i]!=label[k]
    negatives_inside = array_ops.tile(masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]) # [batch_size, batch_size]
     
    # semi_hard_negatives[i][j] = pdist_matrix[i][k] if exists negatives_outside, otherwise negatives_inside
    semi_hard_negatives = array_ops.where(mask_final, negatives_outside, negatives_inside) # [batch_size, batch_size]
    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives) # [batch_size, batch_size]

    # mask_positives[i][j] = 1 if label[i]==label[j], and i!=j
    mask_positives = math_ops.cast(adjacency, dtype=dtypes.float32) - array_ops.diag(array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    # in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    triplet_loss = math_ops.truediv(
            math_ops.reduce_sum(math_ops.maximum(math_ops.multiply(loss_mat, mask_positives), 0.0)),
            num_positives,
            name='triplet_semihard_loss') # hinge
    return triplet_loss

def npairs_loss_hash(labels, embeddings_anchor, embeddings_positive, objective, similarity_func, reg_lambda=0.002):
    """Computes the npairs loss with objective
    similarity base
    Args:
        labels - 1D tensor [batch_size/2],
            tf.int32
        embeddings_anchor - 2D tensor [batch_size/2, embedding_dim]
            embedding vectors for anchor images
        embeddings_positive - 2D tensor [batch_size/2, embedding_dim]
            embedding vectors for positive images
        objective -  2D tensor [batch_size/2, embedding_dim]
            should be binary(0 or 1)
        similarity_func - func 
            args : 
                anc - 2D tensor [ndata, embedding_dim]
                pos - 2D tensor [ndata, embedding_dim]
                obj - 2D tensor [ndata, embedding_dim]
                    which is binary
            return :
                2D tensor [ndata, ndata] 
        reg_lambda - float for  L2 regularization term of embedding vectors
    Returns:
        npairs_loss: tf.float32 scalar.
    """
    reg_anchor = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
    reg_positive = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
    l2loss = math_ops.multiply(0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

    similarity_matrix = similarity_func(anc=embeddings_anchor, pos=embeddings_positive, obj=objective) # [batch_size/2, batch_size/2]
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    labels_remapped = math_ops.to_float(math_ops.equal(labels, array_ops.transpose(labels)))
    labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keep_dims=True)

    # Add the softmax loss.
    xent_loss = nn.softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=labels_remapped)
    xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')
    return l2loss + xent_loss

PAIRWISE_DISTANCE_WITH_OBJECTIVE_DICT={
    'h1' : pairwise_distance_w_obj1
    }

PAIRWISE_SIMILARITY_WITH_OBJECTIVE_DICT={
    'h1' : pairwise_similarity_w_obj1,
    'h2' : pairwise_similarity_w_obj2
    }


