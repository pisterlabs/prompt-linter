import argparse
from relational_embedder.evaluator import coherence_evaluator

if __name__ == "__main__":
    print("Coherence evaluation")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_table', help='path to ground truth table')
    parser.add_argument('--query_entity', help='attribute from where to draw entities to query')
    parser.add_argument('--we_model_path', help='path to we model')
    parser.add_argument('--rel_emb_path', default='row_and_col', help='path to relational_embedding model')
    parser.add_argument('--output_path', help='path to output results')
    parser.add_argument('--num_queries', help='number of queries to emit')
    parser.add_argument('--ranking_size', type=int, default=10)
    parser.add_argument('--hubness_path', default=None, help='File storing word hubness info')

    args = parser.parse_args()

    coherence_evaluator.main(args)
