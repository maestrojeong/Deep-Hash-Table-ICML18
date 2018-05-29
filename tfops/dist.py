from tensorflow.python.framework import dtypes, ops, sparse_tensor, tensor_shape
from tensorflow.python.ops import array_ops, control_flow_ops, logging_ops, math_ops,\
                                  nn, script_ops, sparse_ops
from tensorflow.python.summary import summary
import tensorflow as tf

def deviation_from_kth_element(feature, k):
    '''
    Args:
        feature - 2-D tensor of size [number of data, feature dimensions]
        k - int
            number of bits to be activated
            k < feature dimensions
    Return:
        deviation : 2-D Tensor of size [number of data, feature dimensions] 
           deviation from k th element 
    '''
    feature_top_k = tf.nn.top_k(feature, k=k+1)[0] # [number of data, k+1]

    rho = tf.stop_gradient(tf.add(feature_top_k[:,k-1], feature_top_k[:,k])/2) # [number of data]
    rho_r = tf.reshape(rho, [-1,1]) # [number of data, 1] 

    deviation = tf.subtract(feature, rho_r) # [number of data, feature dimensions]
    return deviation


def pairwise_distance_euclid_v2(feature1, feature2):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
    	feature1 - 2-D Tensor of size [number of data1, feature dimension].
    	feature2 - 2-D Tensor of size [number of data2, feature dimension].
    Returns:
    	pairwise_distances - 2-D Tensor of size [number of data1, number of data2].
    """
    pairwise_distances = math_ops.add(
    	math_ops.reduce_sum(
	    math_ops.square(feature1),
	    axis=[1],
	    keep_dims=True),
        math_ops.reduce_sum(
	    math_ops.square(
	        array_ops.transpose(feature2)),
	        axis=[0],
	        keep_dims=True)) - 2.0 * math_ops.matmul(
	    feature1, array_ops.transpose(feature2))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances = math_ops.maximum(pairwise_distances, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances, 0.0)

    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))
    return pairwise_distances

def pairwise_distance_euclid(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
    	feature - 2-D Tensor of size [number of data, feature dimension].
    	squared - Boolean, whether or not to square the pairwise distances.
    Returns:
    	pairwise_distances - 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
    	math_ops.reduce_sum(
	    math_ops.square(feature),
	    axis=[1],
	    keep_dims=True),
        math_ops.reduce_sum(
	    math_ops.square(
	        array_ops.transpose(feature)),
	        axis=[0],
	        keep_dims=True)) - 2.0 * math_ops.matmul(
	    feature, array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
        data : float, 2D tensor [n, m]
        mask : bool, 2D tensor [n, m]
        dim : int, the dimension over which to compute the maximum.
    Returns:
        masked_maximums : 2D Tensor
            dim=0 => [n,1]
            dim=1 => [1,m]
        get maximum among mask=1
    """
    axis_minimums = math_ops.reduce_min(data, dim, keep_dims=True)
    masked_maximums = math_ops.reduce_max(math_ops.multiply(data - axis_minimums, mask), dim, keep_dims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
        data : float, 2D tensor [n, m]
        mask : bool, 2D tensor [n, m]
        dim : int, the dimension over which to compute the minimum.
    Returns:
        masked_minimums : 2D Tensor
            dim=0 => [n,1]
            dim=1 => [1,m]
        get minimum among mask=1
    """
    axis_maximums = math_ops.reduce_max(data, dim, keep_dims=True) # [n, 1] or [1, m]
    masked_minimums = math_ops.reduce_min(math_ops.multiply(data - axis_maximums, mask), dim, keep_dims=True) + axis_maximums # [n, 1] or [1, m]
    return masked_minimums

def triplet_semihard_loss(labels, embeddings, pairwise_distance, margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.
    Args:
	labels - 1-D tensor [batch_size] as tf.int32
                multiclass integer labels.
	embeddings - 2-D tensor [batch_size, feature dimensions] as tf.float32
                     `Tensor` of embedding vectors. Embeddings should be l2 normalized.
        pairwise_distance - func 
                            with argus 2D tensor [number of data, feature dims]
                                 return 2D tensor [number of data, number of data] 
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
    pdist_matrix = pairwise_distance(embeddings) # [batch_size, batch_size]
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

def npairs_loss(labels, embeddings_anchor, embeddings_positive,
                reg_lambda=0.002, print_losses=False):
    """Computes the npairs loss.
    Npairs loss expects paired data where a pair is composed of samples from the
    same labels and each pairs in the minibatch have different labels. The loss
    has two components. The first component is the L2 regularizer on the
    embedding vectors. The second component is the sum of cross entropy loss
    which takes each row of the pair-wise similarity matrix as logits and
    the remapped one-hot labels as labels.
    See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

    Args:
        labels: 1-D tf.int32 `Tensor` of shape [batch_size/2].
        embeddings_anchor: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
            embedding vectors for the anchor images. Embeddings should not be
            l2 normalized.
        embeddings_positive: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
            embedding vectors for the positive images. Embeddings should not be
            l2 normalized.
        reg_lambda: Float. L2 regularization term on the embedding vectors.
        print_losses: Boolean. Option to print the xent and l2loss.
    Returns:
        npairs_loss: tf.float32 scalar.
    """
    # Add the regularizer on the embedding.
    reg_anchor = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
    reg_positive = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
    l2loss = math_ops.multiply(0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

    # Get per pair similarities.
    similarity_matrix = math_ops.matmul(embeddings_anchor, embeddings_positive, transpose_a=False, transpose_b=True) # [batch_size/2, batch_size/2]

    # Reshape [batch_size/2] label tensor to a [batch_size/2, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # labels_remapped[i][j] = 1 if label[i] == label[j] otherwise 0
    labels_remapped = math_ops.to_float(math_ops.equal(labels, array_ops.transpose(labels))) # [batch_size/2, batch_size/2] 
    labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keep_dims=True)

    # Add the softmax loss.
    xent_loss = nn.softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=labels_remapped)
    xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')

    if print_losses:
        xent_loss = logging_ops.Print(xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])
    return l2loss + xent_loss
