#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    char param;
    int n, e, n1, n2, nNodes, nStates,
            nEdges, 
            *edges, *y;
    double *w1, *w2, *Z, *b1, *b2, logPot, pot;
    
    /* Input */
    param = *(mxChar*)mxGetData(prhs[0]);
    w1 = mxGetPr(prhs[1]);
    w2 = mxGetPr(prhs[2]);
    edges = (int*)mxGetPr(prhs[3]);
    
    /* Compute Sizes */
    nNodes = mxGetDimensions(prhs[1])[0];
    nStates = mxGetDimensions(prhs[1])[1]+1;
    nEdges = mxGetDimensions(prhs[3])[0];
    
    /*printf("Computed Sizes: %d %d %d %d %d %d %d\n", nStates, nNodes, nEdges2, nEdges3, nEdges4, nEdges5, nEdges6);*/
    
    /* Output */
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nNodes, nStates-1, mxREAL);
    /*printf("%c\n",param);*/
    switch (param) {
        case 'C':
        case 'I':
        case 'S':
            /* printf("S\n"); */
            plhs[2] = mxCreateDoubleMatrix(nEdges, 1, mxREAL);
            break;
        case 'P':
            /* printf("P\n"); */
            plhs[2] = mxCreateDoubleMatrix(nStates, nEdges, mxREAL);
            break;
        case 'F':
            /* printf("F\n"); */
            plhs[2] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[2]), mxGetDimensions(prhs[2]), mxDOUBLE_CLASS, mxREAL);
    }
    
    Z = mxGetPr(plhs[0]);
    b1 = mxGetPr(plhs[1]);
    b2 = mxGetPr(plhs[2]);
    
    /* Initialize */
    *Z = 0;
    y = mxCalloc(nNodes, sizeof(int));
    while(1) {
        /* Compute logPot */
        logPot = 0;
        for(n=0;n < nNodes;n++) {
            if(y[n] < nStates-1)
                logPot += w1[n + nNodes*y[n]];
        }
        for(e=0;e < nEdges;e++) {
            n1 = edges[e]-1;
            n2 = edges[e+nEdges]-1;
            switch(param) {
                case 'C':
                    if(y[n1]==0 && y[n2]==0)
                        logPot += w2[e];
                    break;
                case 'I':
                    if(y[n1]==y[n2])
                        logPot += w2[e];
                    break;
                case 'P':
                    if(y[n1]==y[n2])
                        logPot += w2[y[n1] + nStates*e];
                    break;
                case 'S':
                    if((y[n1]+y[n2]) % 2)
                        logPot += w2[e];
                    break;
                case 'F':
                    logPot += w2[y[n1] + nStates*(y[n2] + nStates*e)];
                    
            }
        }
        
        /* Update Z */
        pot = exp(logPot);
        *Z += pot;
        
        /* Update marginals */
        for(n=0;n < nNodes;n++) {
            if(y[n] < nStates-1)
                b1[n + nNodes*y[n]] += pot;
        }
        for(e=0;e < nEdges;e++) {
            n1 = edges[e]-1;
            n2 = edges[e+nEdges]-1;
            switch(param) {
                case 'C':
                    if(y[n1]==0 && y[n2]==0)
                        b2[e] += pot;
                    break;
                case 'I':
                    if(y[n1]==y[n2])
                        b2[e] += pot;
                    break;
                case 'P':
                    if(y[n1]==y[n2])
                        b2[y[n1] + nStates*e] += pot;
                    break;
                case 'S':
                    if((y[n1]+y[n2]) % 2)
                        b2[e] += pot;
                    break;
                case 'F':
                    b2[y[n1] + nStates*(y[n2] + nStates*e)] += pot;
                    
            }
        }
        
        
        /* Go to next state */
        for(n=0;n < nNodes;n++) {
            if(y[n] < nStates-1) {
                y[n]++;
                break;
            }
            else {
                y[n] = 0;
            }
        }
        
        if(n == nNodes && y[nNodes-1]==0) {
            break;
        }
    }
    
    for(n=0;n < nNodes*(nStates-1);n++)
        b1[n] /= *Z;
    switch(param) {
        case 'C':
        case 'I':
        case 'S':
            for(e=0;e < nEdges;e++)
                b2[e] /= *Z;
            break;
        case 'P':
            for(e=0;e < nEdges*nStates;e++)
                b2[e] /= *Z;
            break;
        case 'F':
            for(e=0;e < nEdges*pow(nStates,2);e++)
                b2[e] /= *Z;
                
    }
    
    mxFree(y);
}