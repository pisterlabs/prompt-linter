import argparse
import argparse_config
import sys
import os

from cohere.nn.dA import dA
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import theano.tensor as T
import logging
import cohere.nn.train as train

def main():
    parser = argparse.ArgumentParser(
       description='Run translation experiments.')

    parser.add_argument('--hidden_size', type=int, 
                        help='size of the hidden layer')
    parser.add_argument('--input', type=str, 
                        help='input data')

    parser.add_argument('--training_size', type=int, 
                        help='amount of training data to use.')

    # parser.add_argument('--iterations', type=int, 
    #                     help='iterations of subgrad')
    parser.add_argument('config', type=str)
    parser.add_argument('label', type=str)


    print >>sys.stderr, open(sys.argv[1]).read()
    argparse_config.read_config_file(parser, sys.argv[1])

    args = parser.parse_args()
    print args
    
    output_dir = os.path.join("Data", args.label)
    data_out = os.path.join(output_dir, "mydata.txt")
    print >>sys.stderr, data_out

    # Set up logging.
    logger = logging.getLogger("nn")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(open(data_out, 'w'))
    logger.addHandler(handler)
    
    formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)


    # Read in training data. 
    sparse_lines = []
    n_columns = 0
    for l in open(args.input):
        if not l.strip(): continue
        sparse_line = map(int, l.split())
        sparse_lines.append(sparse_line)
        n_columns = max(n_columns, max(sparse_line))
        if len(sparse_lines) > args.training_size: break
    
    n_columns = n_columns + 1 

    data = numpy.zeros((len(sparse_lines), n_columns))
    for i, line in enumerate(sparse_lines):
        data[i, line] = 1

    data_pairs = numpy.zeros((len(sparse_lines), 2*n_columns))
    for i in range(len(sparse_lines[:-1])):
        line = sparse_lines[i]
        next_line = numpy.array(sparse_lines[i+1])
        data[i, line] = 1
        data[i, next_line + n_columns] = 1

    # Create a hidden layer. 
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    x = T.matrix('x')
    da1 = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
             n_visible=n_columns, n_hidden=args.hidden_size)

    da2 = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
             n_visible=2*args.hidden_size, n_hidden=10)

    train_data = theano.shared(data, borrow=True)
    train.pre_train(da1, train_data, logger=logger)
    logger.info("DONE Pretraining of first layer")


    logger.info("START Pretraining of first layer2")
    layer1 = theano.function([x], da1.get_hidden_values(x))

    layer1_1_output = layer1(data_pairs[:, :columns])
    layer1_2_output = layer1(data_pairs[:, columns:])

    output = theano.shared(numpy.hstack(layer1_1_output, layer1_2_output),
                           borrow=True)
    
    train.pre_train(da2, output, logger=logger)


if __name__ == "__main__":
    main()
    
