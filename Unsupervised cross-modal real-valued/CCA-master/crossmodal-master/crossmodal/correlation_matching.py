from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSCanonical


def correlation_matching(I_tr, T_tr, I_te, T_te, n_comps):
    """ Learns correlation matching (CM) over I_tr and T_tr
        and applies it to I_tr, T_tr, I_te, T_te
        
        
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
        
        I_tr_cca : np.ndarray [shape=(n_tr, n_comps)]
            image data matrix represetned in correlation space
        
        T_tr_cca : np.ndarray [shape=(n_tr, n_comps)]
            text data matrix represetned in correlation space
        
        I_te_cca : np.ndarray [shape=(n_te, n_comps)]
            image data matrix represetned in correlation space
        
        T_te_cca : np.ndarray [shape=(n_te, n_comps)]
            text data matrix represetned in correlation space
        
        """


    # sclale image and text data
    I_scaler = StandardScaler()
    I_tr = I_scaler.fit_transform(I_tr)
    I_te = I_scaler.transform(I_te)

    T_scaler = StandardScaler()
    T_tr = T_scaler.fit_transform(T_tr)
    T_te = T_scaler.transform(T_te)

    cca = PLSCanonical(n_components=n_comps, scale=False)
    cca.fit(I_tr, T_tr)

    I_tr_cca, T_tr_cca = cca.transform(I_tr, T_tr)
    I_te_cca, T_te_cca = cca.transform(I_te, T_te)

    return I_tr_cca, T_tr_cca, I_te_cca, T_te_cca