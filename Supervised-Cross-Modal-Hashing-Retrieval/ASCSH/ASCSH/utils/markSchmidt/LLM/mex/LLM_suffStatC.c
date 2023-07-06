#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    char param;
    int s, n, e, n1, n2, n3, n4, n5, n6, n7, nNodes, nStates, nSamples,
            nEdges2, nEdges3, nEdges4, nEdges5, nEdges6, nEdges7,
            *edges2, *edges3, *edges4, *edges5, *edges6, *edges7, *y, *samples, sizSS[8];
    double *ss1, *ss2, *ss3, *ss4, *ss5, *ss6, *ss7;
    
    /* Input */
    param = *(mxChar*)mxGetData(prhs[0]);
    samples = (int*)mxGetPr(prhs[1]);
    nStates = mxGetScalar(prhs[2]);
    edges2 = (int*)mxGetPr(prhs[3]);
    edges3 = (int*)mxGetPr(prhs[4]);
    edges4 = (int*)mxGetPr(prhs[5]);
    edges5 = (int*)mxGetPr(prhs[6]);
    edges6 = (int*)mxGetPr(prhs[7]);
    edges7 = (int*)mxGetPr(prhs[8]);
    
    /* Compute sizes */
    nSamples = mxGetDimensions(prhs[1])[0];
    nNodes = mxGetDimensions(prhs[1])[1];
    nEdges2 = mxGetDimensions(prhs[3])[0];
    nEdges3 = mxGetDimensions(prhs[4])[0];
    nEdges4 = mxGetDimensions(prhs[5])[0];
    nEdges5 = mxGetDimensions(prhs[6])[0];
    nEdges6 = mxGetDimensions(prhs[7])[0];
    nEdges7 = mxGetDimensions(prhs[8])[0];
    
    /* Output */
    /*printf("Computed Sizes: %d %d %d %d %d %d %d\n", nStates, nNodes, nEdges2, nEdges3, nEdges4, nEdges5, nEdges6);*/
    plhs[0] = mxCreateDoubleMatrix(nNodes, nStates-1, mxREAL);
    switch (param) {
        case 'C':
        case 'I':
        case 'S':
            plhs[1] = mxCreateDoubleMatrix(nEdges2, 1, mxREAL);
            plhs[2] = mxCreateDoubleMatrix(nEdges3, 1, mxREAL);
            plhs[3] = mxCreateDoubleMatrix(nEdges4, 1, mxREAL);
            plhs[4] = mxCreateDoubleMatrix(nEdges5, 1, mxREAL);
            plhs[5] = mxCreateDoubleMatrix(nEdges6, 1, mxREAL);
            plhs[6] = mxCreateDoubleMatrix(nEdges7, 1, mxREAL);
            break;
        case 'P':
            plhs[1] = mxCreateDoubleMatrix(nStates, nEdges2, mxREAL);
            plhs[2] = mxCreateDoubleMatrix(nStates, nEdges3, mxREAL);
            plhs[3] = mxCreateDoubleMatrix(nStates, nEdges4, mxREAL);
            plhs[4] = mxCreateDoubleMatrix(nStates, nEdges5, mxREAL);
            plhs[5] = mxCreateDoubleMatrix(nStates, nEdges6, mxREAL);
            plhs[6] = mxCreateDoubleMatrix(nStates, nEdges7, mxREAL);
            break;
        case 'F':
            for(s=0;s < 7;s++)
                sizSS[s] = nStates;
            sizSS[7] = nEdges7;
            plhs[6] = mxCreateNumericArray(8, sizSS, mxDOUBLE_CLASS, mxREAL);
            sizSS[6] = nEdges6;
            plhs[5] = mxCreateNumericArray(7, sizSS, mxDOUBLE_CLASS, mxREAL);
            sizSS[5] = nEdges5;
            plhs[4] = mxCreateNumericArray(6, sizSS, mxDOUBLE_CLASS, mxREAL);
            sizSS[4] = nEdges4;
            plhs[3] = mxCreateNumericArray(5, sizSS, mxDOUBLE_CLASS, mxREAL);
            sizSS[3] = nEdges3;
            plhs[2] = mxCreateNumericArray(4, sizSS, mxDOUBLE_CLASS, mxREAL);
            sizSS[2] = nEdges2;
            plhs[1] = mxCreateNumericArray(3, sizSS, mxDOUBLE_CLASS, mxREAL);
            
    }
    
    ss1 = mxGetPr(plhs[0]);
    ss2 = mxGetPr(plhs[1]);
    ss3 = mxGetPr(plhs[2]);
    ss4 = mxGetPr(plhs[3]);
    ss5 = mxGetPr(plhs[4]);
    ss6 = mxGetPr(plhs[5]);
    ss7 = mxGetPr(plhs[6]);
    
    y = mxCalloc(nNodes, sizeof(int));
    
    for(s = 0; s < nSamples; s++) {
        for(n = 0; n < nNodes; n++)
            y[n] = samples[s + nSamples*n]-1;
        
        for(n = 0; n < nNodes; n++) {
            if(y[n] < nStates-1)
                ss1[n + nNodes*y[n]] += 1;
        }
        for(e=0;e < nEdges2;e++) {
            n1 = edges2[e]-1;
            n2 = edges2[e+nEdges2]-1;
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
        for(e=0;e < nEdges3;e++) {
            n1 = edges3[e]-1;
            n2 = edges3[e+nEdges3]-1;
            n3 = edges3[e+nEdges3+nEdges3]-1;
            switch(param) {
                case 'C':
                    if(y[n1]==0 && y[n2]==0 && y[n3]==0)
                        ss3[e] += 1;
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3])
                        ss3[e] += 1;
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3])
                        ss3[y[n1] + nStates*e] += 1;
                    break;
                case 'S':
                    if((1+y[n1]+y[n2]+y[n3]) % 2)
                        ss3[e] += 1;
                    break;
                case 'F':
                    ss3[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*e))] += 1;
                    
            }
        }
        for(e=0;e < nEdges4;e++) {
            n1 = edges4[e]-1;
            n2 = edges4[e+nEdges4]-1;
            n3 = edges4[e+nEdges4+nEdges4]-1;
            n4 = edges4[e+nEdges4+nEdges4+nEdges4]-1;
            switch(param) {
                case 'C':
                    if(y[n1]==0 && y[n2]==0 && y[n3]==0 && y[n4]==0)
                        ss4[e] += 1;
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4])
                        ss4[e] += 1;
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4])
                        ss4[y[n1] + nStates*e] += 1;
                    break;
                case 'S':
                    if((y[n1]+y[n2]+y[n3]+y[n4]) % 2)
                        ss4[e] += 1;
                    break;
                case 'F':
                    ss4[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*(y[n4] + nStates*e)))] += 1;
            }
        }
        for(e=0;e < nEdges5;e++) {
            n1 = edges5[e]-1;
            n2 = edges5[e+nEdges5]-1;
            n3 = edges5[e+2*nEdges5]-1;
            n4 = edges5[e+3*nEdges5]-1;
            n5 = edges5[e+4*nEdges5]-1;
            switch(param) {
                case 'C':
                    if(y[n1]==0 && y[n2]==0 && y[n3]==0 && y[n4]==0 && y[n5]==0)
                        ss5[e] += 1;
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5])
                        ss5[e] += 1;
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5])
                        ss5[y[n1] + nStates*e] += 1;
                    break;
                case 'S':
                    if((1+y[n1]+y[n2]+y[n3]+y[n4]+y[n5]) % 2)
                        ss5[e] += 1;
                    break;
                case 'F':
                    ss5[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*(y[n4] + nStates*(y[n5] + nStates*e))))] += 1;
            }
        }
        for(e=0;e < nEdges6;e++) {
            n1 = edges6[e]-1;
            n2 = edges6[e+nEdges6]-1;
            n3 = edges6[e+2*nEdges6]-1;
            n4 = edges6[e+3*nEdges6]-1;
            n5 = edges6[e+4*nEdges6]-1;
            n6 = edges6[e+5*nEdges6]-1;
            switch(param) {
                case 'C':
                    if(y[n1]==0 && y[n2]==0 && y[n3]==0 && y[n4]==0 && y[n5]==0 && y[n6]==0)
                        ss6[e] += 1;
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5] && y[n5]==y[n6])
                        ss6[e] += 1;
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5] && y[n5]==y[n6])
                        ss6[y[n1] + nStates*e] += 1;
                    break;
                case 'S':
                    if((y[n1]+y[n2]+y[n3]+y[n4]+y[n5]+y[n6]) % 2)
                        ss6[e] += 1;
                    break;
                case 'F':
                   ss6[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*(y[n4] + nStates*(y[n5] + nStates*(y[n6] +nStates*e)))))] += 1;
            }
        }
        for(e=0;e < nEdges7;e++) {
            n1 = edges7[e]-1;
            n2 = edges7[e+nEdges7]-1;
            n3 = edges7[e+2*nEdges7]-1;
            n4 = edges7[e+3*nEdges7]-1;
            n5 = edges7[e+4*nEdges7]-1;
            n6 = edges7[e+5*nEdges7]-1;
            n7 = edges7[e+6*nEdges7]-1;
            switch(param) {
                case 'C':
                    if(y[n1]==0 && y[n2]==0 && y[n3]==0 && y[n4]==0 && y[n5]==0 && y[n6]==0 && y[n7]==0)
                        ss7[e] += 1;
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5] && y[n5]==y[n6] && y[n6]==y[n7])
                        ss7[e] += 1;
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5] && y[n5]==y[n6] && y[n6]==y[n7])
                       ss7[y[n1] + nStates*e] += 1;
                    break;
                case 'S':
                    if((1+y[n1]+y[n2]+y[n3]+y[n4]+y[n5]+y[n6]+y[n7]) % 2)
                        ss7[e] += 1;
                    break;
                case 'F':
                    ss7[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*(y[n4] + nStates*(y[n5] + nStates*(y[n6] + nStates*(y[n7] + nStates*e))))))] += 1;
            }
        }
        
    }
    
    for(n=0;n < nNodes*(nStates-1);n++)
        ss1[n] /= nSamples;
    switch(param) {
        case 'C':
        case 'I':
        case 'S':
            for(e=0;e < nEdges2;e++)
                ss2[e] /= nSamples;
            for(e=0;e < nEdges3;e++)
                ss3[e] /= nSamples;
            for(e=0;e < nEdges4;e++)
                ss4[e] /= nSamples;
            for(e=0;e < nEdges5;e++)
                ss5[e] /= nSamples;
            for(e=0;e < nEdges6;e++)
                ss6[e] /= nSamples;
            for(e=0;e < nEdges7;e++)
                ss7[e] /= nSamples;
            break;
        case 'P':
            for(e=0;e < nEdges2*nStates;e++)
                ss2[e] /= nSamples;
            for(e=0;e < nEdges3*nStates;e++)
                ss3[e] /= nSamples;
            for(e=0;e < nEdges4*nStates;e++)
                ss4[e] /= nSamples;
            for(e=0;e < nEdges5*nStates;e++)
                ss5[e] /= nSamples;
            for(e=0;e < nEdges6*nStates;e++)
                ss6[e] /= nSamples;
            for(e=0;e < nEdges7*nStates;e++)
                ss7[e] /= nSamples;
            break;
        case 'F':
            for(e=0;e < nEdges2*pow(nStates, 2);e++)
                ss2[e] /= nSamples;
            for(e=0;e < nEdges3*pow(nStates, 3);e++)
                ss3[e] /= nSamples;
            for(e=0;e < nEdges4*pow(nStates, 4);e++)
                ss4[e] /= nSamples;
            for(e=0;e < nEdges5*pow(nStates, 5);e++)
                ss5[e] /= nSamples;
            for(e=0;e < nEdges6*pow(nStates, 6);e++)
                ss6[e] /= nSamples;
            for(e=0;e < nEdges7*pow(nStates, 7);e++)
                ss7[e] /= nSamples;
            
    }
    
    mxFree(y);
}