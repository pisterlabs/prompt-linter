import argparse
import os
import pickle

from utils import load_data, load_model

from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.evaluation.measures import CoherenceCV, InvertedRBO, TopicDiversity

from knowledge_distillation import get_posteriors, CTMDatasetPosteriors, StudentZeroShotTM



parser = argparse.ArgumentParser()
parser.add_argument("--teacher_model", type=str, required=True)
parser.add_argument("--mode", type=str, choices=['text', 'image'], required=True) 
parser.add_argument("--tries", type=int, default=5, required=False)
parser.add_argument("--use_priors", dest="use_priors", action='store_true')
parser.add_argument("--use_posteriors", dest="use_posteriors", action='store_true')
parser.add_argument("--beta", type=float, default=1.0, required=False)
args = parser.parse_args()

model_type, n_topics = os.path.basename(args.teacher_model).split("_")[:2]

if args.mode == 'image':
    training_dataset = pickle.load(open(os.path.join(args.teacher_model, "image_training_dataset.pkl"), "rb"))
else:
    training_dataset = pickle.load(open(os.path.join(args.teacher_model, "text_training_dataset.pkl"), "rb"))

preprocessed_split = pickle.load(open(os.path.join(args.teacher_model, "preprocessed_split.pkl"), "rb"))


for t in range(args.tries):
    if args.use_posteriors:
        model_dir = "models/joint_%s_student_posteriors/%s_%s_%s_%s/" %(args.mode, model_type, n_topics, args.beta, t)
    elif args.use_priors:
        model_dir = "models/joint_%s_student_priors/%s_%s_%s_%s/" %(args.mode, model_type, n_topics, args.beta, t)
    else:
        model_dir = "models/joint_%s_student/%s_%s_%s_%s/" %(args.mode, model_type, n_topics, args.beta, )
    os.makedirs(model_dir)

    log = open(os.path.join(model_dir,"log.txt"), "w")
    print("Teacher model: %s" %os.path.abspath(args.teacher_model), file=log)

    loss_weights = {"beta":args.beta}
    
    if args.use_posteriors:
        print("Using posteriors")
        teacher_dataset = pickle.load(open(os.path.join(args.teacher_model, "training_dataset.pkl"), "rb"))
        posterior_variance, posterior_mean, posterior_log_variance = get_posteriors(teacher_dataset, args.teacher_model, contextual_size=1024)


        training_dataset = CTMDatasetPosteriors(X_contextual = training_dataset.X_contextual,
                                                X_bow = training_dataset.X_bow,
                                                idx2token = training_dataset.idx2token,
                                                posterior_variance = posterior_variance,
                                                posterior_mean = posterior_mean,
                                                posterior_log_variance = posterior_log_variance)
        
        sctm = StudentZeroShotTM(bow_size=len(training_dataset.idx2token), 
                                 contextual_size=512, 
                                 n_components=int(n_topics), 
                                 model_type=model_type, 
                                 learn_priors=False,
                                 loss_weights=loss_weights)
                                                             
        
    elif args.use_priors:
        tp = pickle.load(open(os.path.join(args.text_model, "tp.pkl"), "rb"))
        tctm = load_model(args.teacher_model, len(tp.vocab))
        sctm = ZeroShotTM(bow_size=len(training_dataset.idx2token), contextual_size=512, n_components=int(n_topics), model_type=model_type, learn_priors=False, loss_weights=loss_weights)
        sctm.model.prior_mean = tctm.model.prior_mean.detach()
        sctm.model.prior_variance = tctm.model.prior_variance.detach()
    
    else:
        sctm = ZeroShotTM(bow_size=len(training_dataset.idx2token), contextual_size=512, n_components=int(n_topics), model_type=model_type, loss_weights=loss_weights)
        

    sctm.fit(training_dataset, save_dir=model_dir, verbose=True)


    topics = sctm.get_topic_lists(25)
    irbo = InvertedRBO(topics=topics)
    td = TopicDiversity(topics=topics)   
    cv = CoherenceCV(texts=preprocessed_split, topics=topics)

    print("IRBO: %2.4f" %irbo.score(), file=log)
    print("Diversity: %2.4f" %td.score(), file=log)
    print("Coherence: %2.4f" %cv.score(), file=log)
    log.close()

