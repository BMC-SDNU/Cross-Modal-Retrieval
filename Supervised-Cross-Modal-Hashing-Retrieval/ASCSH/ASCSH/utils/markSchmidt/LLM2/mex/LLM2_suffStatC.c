#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    char param;
    int s, n, e, n1, n2, nNodes, nStates, nSamples,
            nEdges,
            *edges, *y, *samples, sizSS[3];
    double *ss1, *ss2;
    
    /* Input */
    param = *(mxChar*)mxGetData(prhs[0]);
    samples = (int*)mxGetPr(prhs[1]);
    nStates = mxGetScalar(prhs[2]);
    edges = (int*)mxGetPr(prhs[3]);
    
    /* Compute sizes */
    nSamples = mxGetDimensions(prhs[1])[0];
    nNodes = mxGetDimensions(prhs[1])[1];
    nEdges = mxGetDimensions(prhs[3])[0];
    
    /* Output */
    /*printf("Computed Sizes: %d %d %d %d %d %d %d\n", nStates, nNodes, nEdges2, nEdges3, nEdges4, nEdges5, nEdges6);*/
    plhs[0] = mxCreateDoubleMatrix(nNodes, nStates-1, mxREAL);
    switch (param) {
        case 'C':
        case 'I':
        case 'S':
            plhs[1] = mxCreateDoubleMatrix(nEdges, 1, mxREAL);
            break;
        case 'P':
            plhs[1] = mxCreateDoubleMatrix(nStates, nEdges, mxREAL);
            break;
        case 'F':
            sizSS[0] = nStates;
            sizSS[1] = nStates;
            sizSS[2] = nEdges;
            plhs[1] = mxCreateNumericArray(3, sizSS, mxDOUBLE_CLASS, mxREAL);
            
    }
    
    ss1 = mxGetPr(plhs[0]);
    ss2 = mxGetPr(plhs[1]);
    
    y = mxCalloc(nNodes, sizeof(int));
    
    for(s = 0; s < nSamples; s++) {
        for(n = 0; n < nNodes; n++)
            y[n] = samples[s + nSamples*n]-1;
        
        for(n = 0; n < nNodes; n++) {
            if(y[n] < nStates-1)
                ss1[n + nNodes*y[n]] += 1;
        }
        for(e=0;e < nEdges;e++) {
            n1 = edges[e]-1;
            n2 = edges[e+nEdges]-1;
            switch(param) {
                case 'C':
                    if(y[n1]==0 && y[n2]==0)
                        ss2[e] += 1;
                    break;
                case 'I':
                    if(y[n1]==y[n2])
                        ss2[e] += 1;
                    break;
                case 'P':
                    if(y[n1]==y[n2])
                        ss2[y[n1] + nStates*e] += 1;
                    break;
                case 'S':
                    if((y[n1]+y[n2]) % 2)
                        ss2[e] += 1;
                    break;
                case 'F':
                    ss2[y[n1] + nStates*(y[n2] + nStates*e)] += 1;
                    
            }
        }
        
    }
    
    for(n=0;n < nNodes*(nStates-1);n++)
        ss1[n] /= nSamples;
    switch(param) {
        case 'C':
        case 'I':
        case 'S':
            for(e=0;e < nEdges;e++)
                ss2[e] /= nSamples;
            break;
        case 'P':
            for(e=0;e < nEdges*nStates;e++)
                ss2[e] /= nSamples;
            break;
        case 'F':
            for(e=0;e < nEdges*pow(nStates, 2);e++)
                ss2[e] /= nSamples;
            
    }
    
    mxFree(y);
}