from sklearn.linear_model import LogisticRegression

def _do_classifier(X_tr, X_te, train_truth):
    
    lr_classifier = LogisticRegression(penalty='l2', dual=False, tol=0.01, C=30, fit_intercept=True, intercept_scaling=1)
    lr_classifier.fit(X_tr, train_truth)

    X_tr_lr = lr_classifier.predict_proba(X_tr)
    X_te_lr = lr_classifier.predict_proba(X_te)

    return X_tr_lr, X_te_lr

def semantic_matching(I_tr, T_tr, I_te, T_te, I_truth, T_truth):
    """ Learns semantic matching (CM) over I_tr, respectively T_tr,
        and applies it to I_tr and I_te, and, respectively, T_tr, T_te
        
        
        Parameters
        ----------
        
        I_tr: np.ndarray [shape=(n_tr, d_I)]
        image data matrix for training
        
        T_tr: np.ndarray [shape=(n_tr, d_T)]
        text data matrix for training
        
        I_te: np.ndarray [shape=(n_te, d_I)]
        image data matrix for testing
        
        T_te: np.ndarray [shape=(n_te, d_T)]
        text data matrix for testing
        
        n_comps: int > 0 [scalar]
        number of canonical componens to use
        
        Returns
        -------
        
        I_tr_lr : np.ndarray [shape=(n_tr, n_comps)]
        image data matrix represetned in semantic space
        
        T_tr_lr : np.ndarray [shape=(n_tr, n_comps)]
        text data matrix represetned in semantic space
        
        I_te_lr : np.ndarray [shape=(n_te, n_comps)]
        image data matrix represetned in semantic space
        
        T_te_lr : np.ndarray [shape=(n_te, n_comps)]
        text data matrix represetned in semantic space
        
        """
    
    I_tr_lr, I_te_lr = _do_classifier(I_tr, I_te, I_truth)
    T_tr_lr, T_te_lr = _do_classifier(T_tr, T_te, T_truth)
    return I_tr_lr, T_tr_lr, I_te_lr, T_te_lr
