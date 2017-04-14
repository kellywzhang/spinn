from functools import partial
import argparse
import itertools

import numpy as np
from spinn import util

from spinn.util.blocks import Reduce, treelstm, HeKaimingInitializer, Linear, bundle
from spinn.util.blocks import Embed, to_gpu
from spinn.util.misc import Args, Vocab

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

# Testing
from spinn import afs_safe_logger
from spinn.data.snli import load_snli_data
import gflags
import os

def get_batch(batch):
    X_batch, transitions_batch, y_batch, num_transitions_batch, example_ids = batch

    # Truncate batch.
    X_batch, transitions_batch = truncate(
        X_batch, transitions_batch, num_transitions_batch)

    return X_batch, transitions_batch, y_batch, num_transitions_batch, example_ids

def truncate(X_batch, transitions_batch, num_transitions_batch):
    # Truncate each batch to max length within the batch.
    X_batch_is_left_padded = (not FLAGS.use_left_padding or sequential_only())
    transitions_batch_is_left_padded = FLAGS.use_left_padding
    max_transitions = np.max(num_transitions_batch)
    seq_length = X_batch.shape[1]

    if X_batch_is_left_padded:
        X_batch = X_batch[:, seq_length - max_transitions:]
    else:
        X_batch = X_batch[:, :max_transitions]

    if transitions_batch_is_left_padded:
        transitions_batch = transitions_batch[:, seq_length - max_transitions:]
    else:
        transitions_batch = transitions_batch[:, :max_transitions]

    return X_batch, transitions_batch

def sequential_only():
    return FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW"


def build_model(data_manager, initial_embeddings, vocab_size, num_classes, FLAGS):
    if data_manager.SENTENCE_PAIR_DATA:
        model_cls = SentencePairModel
        use_sentence_pair = True
    else:
        model_cls = SentenceModel
        use_sentence_pair = False

    return model_cls(model_dim=FLAGS.model_dim,
         word_embedding_dim=FLAGS.word_embedding_dim,
         vocab_size=vocab_size,
         initial_embeddings=initial_embeddings,
         num_classes=num_classes,
         mlp_dim=FLAGS.mlp_dim,
         embedding_keep_rate=FLAGS.embedding_keep_rate,
         classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
         use_sentence_pair=use_sentence_pair,
         use_difference_feature=FLAGS.use_difference_feature,
         use_product_feature=FLAGS.use_product_feature,
         num_mlp_layers=FLAGS.num_mlp_layers,
         mlp_bn=FLAGS.mlp_bn,
        )

class LeftTree(nn.Module):

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
                 mlp_dim=None,
                 embedding_keep_rate=None,
                 use_sentence_pair=False,
                 **kwargs
                ):
        super(LeftTree, self).__init__()

        self.model_dim = model_dim

        args = Args()
        args.size = model_dim
        args.input_dropout_rate = 1. - embedding_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        self.embed = Embed(args.size, vocab.size,
                        vectors=vocab.vectors,
                        )

        # Reduce function for semantic composition.
        self.left_weight = Linear(initializer=HeKaimingInitializer)(args.size, 5 * args.size)
        self.right_weight = Linear(initializer=HeKaimingInitializer)(args.size, 5 * args.size, bias=False)

        mlp_input_dim = model_dim * 2 if use_sentence_pair else model_dim

        self.l0 = nn.Linear(mlp_input_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, mlp_dim)
        self.l2 = nn.Linear(mlp_dim, num_classes)

    def run_tree(self, x):
        batch_size, seq_len, model_dim = x.data.size()

        gates = self.left_weight(x[:,0])
        gates += self.right_weight(x[:,1])

        c_t, h_t = treelstm(c_left=x[:,0], c_right=x[:,1], gates=gates, use_dropout=False, training=self.training)

        # simple no variable length
        for i in range(2, seq_len):
            gates = self.left_weight(c_t)
            gates += self.right_weight(x[:,i])
            c_t, h_t = treelstm(c_left=c_t, c_right=x[:,i], gates=gates, use_dropout=False, training=self.training)

        return h_t


    def run_embed(self, x):
        batch_size, seq_length = x.size()
        
        emb = self.embed(x)
        emb = torch.cat([b.unsqueeze(0) for b in torch.chunk(emb, batch_size, 0)], 0)
        return emb

    def run_mlp(self, h):
        h = self.l0(h)
        h = F.relu(h)
        h = self.l1(h)
        h = F.relu(h)
        h = self.l2(h)
        y = h
        return y

class SoftSPINN(nn.Module):
    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
                 mlp_dim=None,
                 embedding_keep_rate=None,
                 use_sentence_pair=False,
                 **kwargs
                ):
        
        super(SoftSPINN, self).__init__()

        self.model_dim = model_dim

        args = Args()
        args.size = model_dim
        args.input_dropout_rate = 1. - embedding_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings
        
        self.softstack = SoftStack(embedding_dim=args.size,
                                    hidden_size=args.size,
                                    vocab_size=vocab.size,
                                    vocab_vectors=vocab.vectors)

        mlp_input_dim = model_dim * 2 if use_sentence_pair else model_dim

        self.l0 = nn.Linear(mlp_input_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, mlp_dim)
        self.l2 = nn.Linear(mlp_dim, num_classes)

    def run_mlp(self, h):
        h = self.l0(h)
        h = F.relu(h)
        h = self.l1(h)
        h = F.relu(h)
        h = self.l2(h)
        y = h
        return y

class SoftStack(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, vocab_vectors=None):
        super(SoftStack, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Reduce function for semantic composition.
        self.tree_left = nn.Linear(in_features=hidden_size, out_features=5*hidden_size)
        self.tree_right = nn.Linear(in_features=hidden_size, out_features=5*hidden_size)
        # self.tree_left = Linear(initializer=HeKaimingInitializer)(args.size, 5 * args.size)
        # self.tree_left = Linear(initializer=HeKaimingInitializer)(args.size, 5 * args.size, bias=False)
        
        self.embedding = Embed(embedding_dim, vocab_size,
                        vectors=vocab_vectors,
                        )

        self.controller = nn.LSTMCell(input_size=hidden_size*3, hidden_size=hidden_size)
        self.alpha_projector = nn.Linear(in_features=hidden_size, out_features=2)
        
    def readBuffer(self, alpha, B, s_b, pop=False):
        batch_size = B.size(0)
        sequence_len = B.size(1)

        cumsum = Variable(torch.zeros(batch_size).float())
        vector = Variable(torch.zeros(batch_size, self.embedding_dim).float())
        batch_zeros = Variable(torch.zeros(batch_size))
        batch_alpha = Variable(torch.ones(batch_size)) if alpha is None else alpha
        
        # May be more efficient not to loop like this... especially when buffer is empty
        for i in range(sequence_len):

            weights = torch.min(s_b[:, i], torch.max(batch_zeros, batch_alpha - cumsum))  
            vector = torch.add(vector, torch.mul(weights.unsqueeze(1).expand_as(B[:, i]), B[:, i]))
            cumsum = torch.add(cumsum, weights)
            if pop:
                s_b[:, i] = torch.add(s_b[:, i], torch.mul(weights, -1))

            if batch_size <= torch.sum(torch.ge(cumsum, batch_alpha)).data[0]:
                break

        return vector, s_b
    
    def pushStack(self, vector, alpha, V, s):
        V = [vector] + V
        s = [alpha] + s

        return V, s
    
    def readStack(self, alpha, V, s, pop=False):
        batch_size = V[0].size(0)
        embedding_dim = V[0].size(1)
    
        vector1 = Variable(torch.zeros(batch_size, embedding_dim).float())
        vector2 = Variable(torch.zeros(batch_size, embedding_dim).float())

        cumsum = Variable(torch.zeros(batch_size).float())
        batch_zeros = Variable(torch.zeros(batch_size))
        batch_ones = Variable(torch.ones(batch_size))
        batch_alpha = Variable(torch.ones(batch_size)) if alpha is None else alpha

        # May be more efficient not to loop like this... especially when stack is empty
        for i in range(len(V)):
            weights1 = torch.min(s[i], torch.max(batch_zeros, batch_alpha - cumsum))
            weights2 = torch.min(s[i], torch.max(batch_zeros, batch_ones + batch_alpha - cumsum)) - weights1
            cumsum = torch.add(cumsum, weights1+weights2)

            vector1 = torch.add(vector1, torch.mul(weights1.unsqueeze(1).expand_as(V[0]), V[i]))
            vector2 = torch.add(vector2, torch.mul(weights2.unsqueeze(1).expand_as(V[0]), V[i]))

            if pop:
                s[i] = torch.add(s[i], torch.mul(weights1+weights2, -1))

            if batch_size == torch.sum(torch.ge(cumsum, 2)).data[0]:
                break

        return vector1, vector2, s
    
    def init_controller(self, batch_size):
        # h_t, c_t
        state = (Variable(torch.zeros(batch_size, self.hidden_size)), \
                Variable(torch.zeros(batch_size, self.hidden_size)))
        return state
    
    def forward(self, x, timesteps=None):
        beta = 10
        batch_size = x.size(0)
        seq_len = x.size(1)
        if timesteps is None:
            time = 2*seq_len
        
        # initialize buffer
        B = self.run_embed(x)
        s_b = torch.gt(x, 0).float()
        
        # initialize stack
        V = [Variable(torch.zeros(batch_size, self.hidden_size))]
        s = [Variable(torch.zeros(batch_size))]
        controller_state = self.init_controller(batch_size)
        
        # Remember stack operations; remember indicies of x; reverse dictionary
        self.memory = {}

        # LSTM controller timesteps
        for t in range(time):
            # Read from stack and buffer
            x_b, _ = self.readBuffer(alpha=None, B=B, s_b=s_b, pop=False)
            x_1, x_2, _ = self.readStack(alpha=None, V=V, s=s, pop=False)
            x_t = torch.cat([x_b, x_1, x_2], 1)
            
            # Get alphas from controller
            controller_state = self.controller(x_t, controller_state)
            hidden_state = controller_state[0]
            alphas = F.softmax(torch.mul(self.alpha_projector(hidden_state), beta))
            alpha_r = alphas[:, 0]
            alpha_s = alphas[:, 1]
            # print(alpha_s)
            
            # Read from stack, reduce (treelstm), and push onto stack
            stack_reduce1, stack_reduce2, s = self.readStack(alpha_r, V, s, pop=True)
            tree_state = self.run_tree(stack_reduce1, stack_reduce2)
            hidden_state_tree = tree_state[0]
            V, s = self.pushStack(vector=hidden_state_tree, alpha=alpha_r, V=V, s=s)
            
            # Shift from buffer and push onto stack
            buffer_shift, s_b = self.readBuffer(alpha_s, B, s_b, pop=True)
            V, s = self.pushStack(vector=buffer_shift, alpha=alpha_s, V=V, s=s)
        
        # print(alpha_s)
        # print(s[:3])
        print(s_b)
        # Pop from top of stack with strength 1 as final sentence representation
        x_1, x_2, _ = self.readStack(alpha=None, V=V, s=s, pop=False)
        return x_1

    def run_embed(self, x):
        batch_size, seq_length = x.size()

        emb = self.embedding(x)
        emb = torch.cat([b.unsqueeze(0) for b in torch.chunk(emb, batch_size, 0)], 0)

        return emb

    def run_tree(self, v1, v2, use_dropout=False):
        gates = self.tree_left(v1)
        gates += self.tree_right(v2)
        c_t, h_t = treelstm(c_left=v1, c_right=v2, gates=gates, use_dropout=use_dropout, training=self.training)
        return (h_t, c_t)

class SentencePairModel(LeftTree):

    def build_example(self, sentences):
        batch_size = sentences.shape[0]

        # Build Tokens
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        return to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))

    def forward(self, sentences, transitions_batch=None, y_batch=None, use_internal_parser=False,
                validate_transitions=False, **kwargs):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = self.build_example(sentences)

        emb = self.run_embed(x)

        hh = self.run_tree(emb)

        h = torch.cat([hh[:batch_size], hh[batch_size:]], 1)
        output = self.run_mlp(h)

        return output


class SentenceModelTree(LeftTree):

    def build_example(self, sentences, transitions):
        return to_gpu(Variable(torch.from_numpy(sentences), volatile=not self.training))

    def forward(self, sentences, transitions, y_batch=None, **kwargs):
        # Build Tokens
        x = self.build_example(sentences, transitions)

        print(kwargs['num_transitions_batch'])

        emb = self.run_embed(x)

        h = self.run_tree(emb)

        output = self.run_mlp(h)

        return output

class SentenceModel(SoftSPINN):

    def build_example(self, sentences, transitions):
        return to_gpu(Variable(torch.from_numpy(sentences), volatile=not self.training))

    def forward(self, sentences, transitions, y_batch=None, **kwargs):
        # Build Tokens
        x = self.build_example(sentences, transitions)

        h = self.softstack(x)

        output = self.run_mlp(h)

        return output


if __name__ == "__main__":
    gflags.DEFINE_integer("model_dim", 8, "")
    gflags.DEFINE_integer("word_embedding_dim", 8, "")

    gflags.DEFINE_enum("model_type", "RNN", ["CBOW", "RNN", "SPINN", "RLSPINN", "RAESPINN", "GENSPINN"], "")
    gflags.DEFINE_integer("batch_size", 32, "SGD minibatch size.")
    gflags.DEFINE_boolean("smart_batching", True, "Organize batches using sequence length.")
    gflags.DEFINE_boolean("use_peano", True, "A mind-blowing sorting key.")
    gflags.DEFINE_float("embedding_keep_rate", 0.9,
        "Used for dropout on transformed embeddings and in the encoder RNN.")

    # MLP settings.
    gflags.DEFINE_integer("mlp_dim", 1024, "Dimension of intermediate MLP layers.")
    gflags.DEFINE_integer("num_mlp_layers", 2, "Number of MLP layers.")
    gflags.DEFINE_boolean("mlp_bn", True, "When True, batch normalization is used between MLP layers.")

    gflags.DEFINE_integer("seq_length", 30, "")
    gflags.DEFINE_string("training_data_path", "", "")
    gflags.DEFINE_boolean("lowercase", True, "")
    gflags.DEFINE_string("log_path", "", "")
    gflags.DEFINE_string("experiment_name", "treetest", "")
    gflags.DEFINE_boolean("use_left_padding", True, "Pad transitions only on the LHS.")
    gflags.DEFINE_string("embedding_data_path", "",
        "If set, load GloVe-formatted embeddings from here.")
    FLAGS = gflags.FLAGS
    logger = afs_safe_logger.Logger(os.path.join(FLAGS.log_path, FLAGS.experiment_name) + ".log")

    args = Args()
    args.size = 20

    data_manager = load_snli_data

    # Load the data.
    raw_training_data, vocabulary = data_manager.load_data(
        FLAGS.training_data_path, FLAGS.lowercase)

    # Load the eval data.
    raw_eval_sets = []

    # Prepare the vocabulary.
    if not vocabulary:
        logger.Log("In open vocabulary mode. Using loaded embeddings without fine-tuning.")
        train_embeddings = False
        vocabulary = util.BuildVocabulary(
            raw_training_data, raw_eval_sets, FLAGS.embedding_data_path, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
    else:
        logger.Log("In fixed vocabulary mode. Training embeddings.")
        train_embeddings = True


    # Trim dataset, convert token sequences to integer sequences, crop, and
    # pad.
    logger.Log("Preprocessing training data.")
    training_data = util.PreprocessDataset(
        raw_training_data, vocabulary, FLAGS.seq_length, data_manager, eval_mode=False, logger=logger,
        sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
        for_rnn=True,
        use_left_padding=FLAGS.use_left_padding)
    training_data_iter = util.MakeTrainingIterator(
        training_data, FLAGS.batch_size, FLAGS.smart_batching, FLAGS.use_peano,
        sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)

    model = SentencePairModel(model_dim=FLAGS.model_dim,
                    word_embedding_dim=FLAGS.word_embedding_dim,
                    vocab_size=len(vocabulary),
                    initial_embeddings=None,
                    num_classes=3,
                    mlp_dim=FLAGS.mlp_dim,
                    embedding_keep_rate=FLAGS.embedding_keep_rate,
                    use_sentence_pair=True)
    print(model)

    # Build log format strings.
    model.train()
    X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = get_batch(training_data_iter.next())
    model(X_batch, y_batch)

