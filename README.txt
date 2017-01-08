Multi-Class Bayesian Logistic Regression Using Boning’s Approximation

Background:

    The following is an implementation of Bayesian Logistic Regression, learned via Variational Inference. Currently it is trained with Automatic Relevance Determination (ARD) to regularize the weights and induce sparsity in the parameters. In order to apply Variational Inference, the log-sum-exp component of the objective function is approximated by Bohning’s Bound. 

    The bulk of the logic to learn the classifier lies in bohning_logistic.py. The data_preprocessor.py file, transforms specific raw datasets (such as those from the scikit-learn package) into the required format for the classifier to ingest. 

    Some possible extensions of this implementation could be to model temporal or structured information, create a mixture of Bayesian Logistic Regressors, and to allow the ability to retrain the parameters after pruning with ARD (for increased accuracy). 

    NOTE: With very high-dimensional datasets (like images) some overflow issues occur, causing the ELBO value to start increasing. Some tricks like using log-sum-exp when performing matrix multiplications could help address this issue.

Usage:
    
    Install the required packages from the requirements file: `pip install -r requirements.txt`.
    
    To train and evaluate the classifier on a dataset: python bohning_logistic.py -d <dataset> [--synth-x-dim SYNTH_X_DIM] [--synth-y-dim SYNTH_Y_DIM] [--synth-x-dep SYNTH_X_DEP [--elbo-thresh ELBO_THRESH] [--prune-threshold PRUNE_THRESHOLD] [--iter ITER] [--a0 A0] [--b0 B0] [--max-iter MAX_ITER] [-v VERBOSE]

    To see an example run of the classifier refer to the ipython file: sample_training_run.ipynb 
