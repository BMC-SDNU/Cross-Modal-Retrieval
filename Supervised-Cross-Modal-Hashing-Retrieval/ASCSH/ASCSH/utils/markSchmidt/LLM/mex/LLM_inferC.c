#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    char param;
    int n, e, n1, n2, n3, n4, n5, n6, n7, nNodes, nStates, sizB[8],
            nEdges2, nEdges3, nEdges4, nEdges5, nEdges6, nEdges7,
            *edges2, *edges3, *edges4, *edges5, *edges6, *edges7, *y;
    double *w1, *w2, *w3, *w4, *w5, *w6, *w7, *Z, *b1, *b2, *b3, *b4, *b5, *b6, *b7, logPot, pot;
    
    /* Input */
    param = *(mxChar*)mxGetData(prhs[0]);
    w1 = mxGetPr(prhs[1]);
    w2 = mxGetPr(prhs[2]);
    w3 = mxGetPr(prhs[3]);
    w4 = mxGetPr(prhs[4]);
    w5 = mxGetPr(prhs[5]);
    w6 = mxGetPr(prhs[6]);
    w7 = mxGetPr(prhs[7]);
    edges2 = (int*)mxGetPr(prhs[8]);
    edges3 = (int*)mxGetPr(prhs[9]);
    edges4 = (int*)mxGetPr(prhs[10]);
    edges5 = (int*)mxGetPr(prhs[11]);
    edges6 = (int*)mxGetPr(prhs[12]);
    edges7 = (int*)mxGetPr(prhs[13]);
    
    /* Compute Sizes */
    nNodes = mxGetDimensions(prhs[1])[0];
    nStates = mxGetDimensions(prhs[1])[1]+1;
    nEdges2 = mxGetDimensions(prhs[8])[0];
    nEdges3 = mxGetDimensions(prhs[9])[0];
    nEdges4 = mxGetDimensions(prhs[10])[0];
    nEdges5 = mxGetDimensions(prhs[11])[0];
    nEdges6 = mxGetDimensions(prhs[12])[0];
    nEdges7 = mxGetDimensions(prhs[13])[0];
    
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
            plhs[2] = mxCreateDoubleMatrix(nEdges2, 1, mxREAL);
            plhs[3] = mxCreateDoubleMatrix(nEdges3, 1, mxREAL);
            plhs[4] = mxCreateDoubleMatrix(nEdges4, 1, mxREAL);
            plhs[5] = mxCreateDoubleMatrix(nEdges5, 1, mxREAL);
            plhs[6] = mxCreateDoubleMatrix(nEdges6, 1, mxREAL);
            plhs[7] = mxCreateDoubleMatrix(nEdges7, 1, mxREAL);
            break;
        case 'P':
            /* printf("P\n"); */
            plhs[2] = mxCreateDoubleMatrix(nStates, nEdges2, mxREAL);
            plhs[3] = mxCreateDoubleMatrix(nStates, nEdges3, mxREAL);
            plhs[4] = mxCreateDoubleMatrix(nStates, nEdges4, mxREAL);
            plhs[5] = mxCreateDoubleMatrix(nStates, nEdges5, mxREAL);
            plhs[6] = mxCreateDoubleMatrix(nStates, nEdges6, mxREAL);
            plhs[7] = mxCreateDoubleMatrix(nStates, nEdges7, mxREAL);
            break;
        case 'F':
            /* printf("F\n"); */
            plhs[2] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[2]), mxGetDimensions(prhs[2]), mxDOUBLE_CLASS, mxREAL);
            plhs[3] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[3]), mxGetDimensions(prhs[3]), mxDOUBLE_CLASS, mxREAL);
            plhs[4] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[4]), mxGetDimensions(prhs[4]), mxDOUBLE_CLASS, mxREAL);
            plhs[5] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[5]), mxGetDimensions(prhs[5]), mxDOUBLE_CLASS, mxREAL);
            plhs[6] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[6]), mxGetDimensions(prhs[6]), mxDOUBLE_CLASS, mxREAL);
            plhs[7] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[7]), mxGetDimensions(prhs[7]), mxDOUBLE_CLASS, mxREAL);
    }
    
    Z = mxGetPr(plhs[0]);
    b1 = mxGetPr(plhs[1]);
    b2 = mxGetPr(plhs[2]);
    b3 = mxGetPr(plhs[3]);
    b4 = mxGetPr(plhs[4]);
    b5 = mxGetPr(plhs[5]);
    b6 = mxGetPr(plhs[6]);
    b7 = mxGetPr(plhs[7]);
    
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
        for(e=0;e < nEdges2;e++) {
            n1 = edges2[e]-1;
            n2 = edges2[e+nEdges2]-1;
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
        for(e=0;e < nEdges3;e++) {
            n1 = edges3[e]-1;
            n2 = edges3[e+nEdges3]-1;
            n3 = edges3[e+nEdges3+nEdges3]-1;
            switch(param) {
                case 'C':
                    if(y[n1]==0 && y[n2]==0 && y[n3]==0)
                        logPot += w3[e];
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3])
                        logPot += w3[e];
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3])
                        logPot += w3[y[n1] + nStates*e];
                    break;
                case 'S':
                    if((1+y[n1]+y[n2]+y[n3]) % 2)
                        logPot += w3[e];
                    break;
                case 'F':
                    logPot += w3[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*e))];
                    
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
                        logPot += w4[e];
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4])
                        logPot += w4[e];
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4])
                        logPot += w4[y[n1] + nStates*e];
                    break;
                case 'S':
                    if((y[n1]+y[n2]+y[n3]+y[n4]) % 2)
                        logPot += w4[e];
                    break;
                case 'F':
                    logPot += w4[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*(y[n4] + nStates*e)))];
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
                        logPot += w5[e];
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5])
                        logPot += w5[e];
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5])
                        logPot += w5[y[n1] + nStates*e];
                    break;
                case 'S':
                    if((1+y[n1]+y[n2]+y[n3]+y[n4]+y[n5]) % 2)
                        logPot += w5[e];
                    break;
                case 'F':
                    logPot += w5[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*(y[n4] + nStates*(y[n5] + nStates*e))))];
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
                        logPot += w6[e];
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5] && y[n5]==y[n6])
                        logPot += w6[e];
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5] && y[n5]==y[n6])
                        logPot += w6[y[n1] + nStates*e];
                    break;
                case 'S':
                    if((y[n1]+y[n2]+y[n3]+y[n4]+y[n5]+y[n6]) % 2)
                        logPot += w6[e];
                    break;
                case 'F':
                    logPot += w6[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*(y[n4] + nStates*(y[n5] + nStates*(y[n6] +nStates*e)))))];
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
                        logPot += w7[e];
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5] && y[n5]==y[n6] && y[n6]==y[n7])
                        logPot += w7[e];
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5] && y[n5]==y[n6] && y[n6]==y[n7])
                        logPot += w7[y[n1] + nStates*e];
                    break;
                case 'S':
                    if((1+y[n1]+y[n2]+y[n3]+y[n4]+y[n5]+y[n6]+y[n7]) % 2)
                        logPot += w7[e];
                    break;
                case 'F':
                    logPot += w7[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*(y[n4] + nStates*(y[n5] + nStates*(y[n6] + nStates*(y[n7] + nStates*e))))))];
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
        for(e=0;e < nEdges2;e++) {
            n1 = edges2[e]-1;
            n2 = edges2[e+nEdges2]-1;
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
        for(e=0;e < nEdges3;e++) {
            n1 = edges3[e]-1;
            n2 = edges3[e+nEdges3]-1;
            n3 = edges3[e+nEdges3+nEdges3]-1;
            switch(param) {
                case 'C':
                    if(y[n1]==0 && y[n2]==0 && y[n3]==0)
                        b3[e] += pot;
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3])
                        b3[e] += pot;
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3])
                        b3[y[n1] + nStates*e] += pot;
                    break;
                case 'S':
                    if((1+y[n1]+y[n2]+y[n3]) % 2)
                        b3[e] += pot;
                    break;
                case 'F':
                    b3[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*e))] += pot;
                    
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
                        b4[e] += pot;
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4])
                        b4[e] += pot;
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4])
                        b4[y[n1] + nStates*e] += pot;
                    break;
                case 'S':
                    if((y[n1]+y[n2]+y[n3]+y[n4]) % 2)
                        b4[e] += pot;
                    break;
                case 'F':
                    b4[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*(y[n4] + nStates*e)))] += pot;
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
                        b5[e] += pot;
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5])
                        b5[e] += pot;
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5])
                        b5[y[n1] + nStates*e] += pot;
                    break;
                case 'S':
                    if((1+y[n1]+y[n2]+y[n3]+y[n4]+y[n5]) % 2)
                        b5[e] += pot;
                    break;
                case 'F':
                    b5[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*(y[n4] + nStates*(y[n5] + nStates*e))))] += pot;
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
                        b6[e] += pot;
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5] && y[n5]==y[n6])
                        b6[e] += pot;
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5] && y[n5]==y[n6])
                        b6[y[n1] + nStates*e] += pot;
                    break;
                case 'S':
                    if((y[n1]+y[n2]+y[n3]+y[n4]+y[n5]+y[n6]) % 2)
                        b6[e] += pot;
                    break;
                case 'F':
                   b6[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*(y[n4] + nStates*(y[n5] + nStates*(y[n6] +nStates*e)))))] += pot;
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
                        b7[e] += pot;
                    break;
                case 'I':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5] && y[n5]==y[n6] && y[n6]==y[n7])
                        b7[e] += pot;
                    break;
                case 'P':
                    if(y[n1]==y[n2] && y[n2]==y[n3] && y[n3]==y[n4] && y[n4]==y[n5] && y[n5]==y[n6] && y[n6]==y[n7])
                       b7[y[n1] + nStates*e] += pot;
                    break;
                case 'S':
                    if((1+y[n1]+y[n2]+y[n3]+y[n4]+y[n5]+y[n6]+y[n7]) % 2)
                        b7[e] += pot;
                    break;
                case 'F':
                    b7[y[n1] + nStates*(y[n2] + nStates*(y[n3] + nStates*(y[n4] + nStates*(y[n5] + nStates*(y[n6] + nStates*(y[n7] + nStates*e))))))] += pot;
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
            for(e=0;e < nEdges2;e++)
                b2[e] /= *Z;
            for(e=0;e < nEdges3;e++)
                b3[e] /= *Z;
            for(e=0;e < nEdges4;e++)
                b4[e] /= *Z;
            for(e=0;e < nEdges5;e++)
                b5[e] /= *Z;
            for(e=0;e < nEdges6;e++)
                b6[e] /= *Z;
            for(e=0;e < nEdges7;e++)
                b7[e] /= *Z;
            break;
        case 'P':
            for(e=0;e < nEdges2*nStates;e++)
                b2[e] /= *Z;
            for(e=0;e < nEdges3*nStates;e++)
                b3[e] /= *Z;
            for(e=0;e < nEdges4*nStates;e++)
                b4[e] /= *Z;
            for(e=0;e < nEdges5*nStates;e++)
                b5[e] /= *Z;
            for(e=0;e < nEdges6*nStates;e++)
                b6[e] /= *Z;
            for(e=0;e < nEdges7*nStates;e++)
                b7[e] /= *Z;
            break;
        case 'F':
            for(e=0;e < nEdges2*pow(nStates,2);e++)
                b2[e] /= *Z;
            for(e=0;e < nEdges3*pow(nStates,3);e++)
                b3[e] /= *Z;
            for(e=0;e < nEdges4*pow(nStates,4);e++)
                b4[e] /= *Z;
            for(e=0;e < nEdges5*pow(nStates,5);e++)
                b5[e] /= *Z;
            for(e=0;e < nEdges6*pow(nStates,6);e++)
                b6[e] /= *Z;
            for(e=0;e < nEdges7*pow(nStates,7);e++)
                b7[e] /= *Z;
                
    }
    
    mxFree(y);
}