#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* variables */
    char param;
    int i, s, n, e, n1, n2, nInstances, nNodes, nStates, nEdges,
            *edges, *y, y1, y2;
    double *yr, *g1, *g2, *logPot, *z, *pseudoNLL, *w1, *w2, *nodeBel;
    
    /* input */
    param = *(mxChar*)mxGetData(prhs[0]);
    y = (int*)mxGetPr(prhs[1]);
    yr = mxGetPr(prhs[2]);
    g1 = mxGetPr(prhs[3]);
    g2 = mxGetPr(prhs[4]);
    edges = (int*)mxGetPr(prhs[5]);
    w1 = mxGetPr(prhs[6]);
    w2 = mxGetPr(prhs[7]);
    
    /* compute sizes */
    nInstances = mxGetDimensions(prhs[1])[0];
    nNodes = mxGetDimensions(prhs[6])[0];
    nStates = mxGetDimensions(prhs[6])[1]+1;
    nEdges = mxGetDimensions(prhs[5])[0];
    
    /* allocate memory */
    logPot = mxCalloc(nStates*nNodes, sizeof(double));
    z = mxCalloc(nNodes, sizeof(double));
    nodeBel = mxCalloc(nStates*nNodes, sizeof(double));
    
    /* output */
    plhs[0] = mxCreateDoubleMatrix(1, 1, 0);
    pseudoNLL = mxGetPr(plhs[0]);
    *pseudoNLL = 0;
    
    /*printf("computed sizes: %d %d %d %d\n",nStates,nNodes,nEdges2,nEdges3);*/
    
    for(i=0;i < nInstances;i++) {
        for(n=0;n < nNodes;n++) {
            for(s=0;s < nStates-1;s++) {
                logPot[s+nStates*n] = w1[n+nNodes*s];
            }
            logPot[nStates-1+nStates*n] = 0;
        }
        for(e = 0;e < nEdges;e++) {
            n1 = edges[e];
            n2 = edges[e+nEdges];
            y1 = y[i + nInstances*n1];
            y2 = y[i + nInstances*n2];
            
            switch (param) {
                case 'C':
                    if (y2==0)
                        logPot[nStates*n1] += w2[e];
                    if (y1==0)
                        logPot[nStates*n2] += w2[e];
                    break;
                case 'I':
                    logPot[y2+nStates*n1] += w2[e];
                    logPot[y1+nStates*n2] += w2[e];
                    break;
                case 'P':
                    logPot[y2+nStates*n1] += w2[y2+nStates*e];
                    logPot[y1+nStates*n2] += w2[y1+nStates*e];
                    break;
                case 'S':
                    logPot[(1+y2) % 2 + nStates*n1] += w2[e];
                    logPot[(1+y1) % 2 + nStates*n2] += w2[e];
                    break;
                case 'F':
                    for(s=0; s<nStates; s++) {
                        logPot[s+nStates*n1] += w2[s+nStates*(y2+nStates*e)];
                        logPot[s+nStates*n2] += w2[y1+nStates*(s+nStates*e)];
                    }
            }
        }
        
        
        /*for(s = 0; s < nStates; s++) {
         * printf("logPot(%d,:) = [", s);
         * for(n = 0;n < nNodes; n++) {
         * printf(" %f", logPot[s+nStates*n]);
         * }
         * printf(" ]\n");
         * }*/
        
        for(n = 0;n < nNodes;n++) {
            z[n] = 0;
            for(s = 0; s < nStates; s++) {
                z[n] += exp(logPot[s+nStates*n]);
            }
            *pseudoNLL -= yr[i]*logPot[y[i+nInstances*n] + nStates*n];
            *pseudoNLL += yr[i]*log(z[n]);
        }
        
        /*printf("pseudoNLL = %f\n",*pseudoNLL);*/
        
        for(n = 0;n < nNodes;n++) {
            for(s = 0; s < nStates; s++) {
                nodeBel[s + nStates*n] = exp(logPot[s + nStates*n] - log(z[n]));
            }
        }
        
        /*for(n = 0;n < nNodes;n++) {
         * for(s = 0; s < nStates; s++) {
         * printf("nodeBel(%d,%d) = %f\n",s+1,n+1,nodeBel[s+nStates*n]);
         * }
         * }*/
        
        for(n = 0;n < nNodes;n++) {
            y1 = y[i + nInstances*n];
            
            if(y1 < nStates-1)
                g1[n + nNodes*y1] -= yr[i]*1;
            
            for(s=0; s < nStates-1; s++)
                g1[n + nNodes*s] += yr[i]*nodeBel[s+nStates*n];
        }
        for(e = 0;e < nEdges;e++) {
            n1 = edges[e];
            n2 = edges[e+nEdges];
            y1 = y[i + nInstances*n1];
            y2 = y[i + nInstances*n2];
            
            switch (param) {
                case 'C':
                    if (y1==0 && y2==0)
                        g2[e] -= yr[i]*2;
                    if (y2==0)
                        g2[e] += yr[i]*nodeBel[nStates*n1];
                    if (y1==0)
                        g2[e] += yr[i]*nodeBel[nStates*n2];
                    break;
                case 'I':
                    if (y1==y2)
                        g2[e] -= yr[i]*2;
                    g2[e] += yr[i]*nodeBel[y2+nStates*n1];
                    g2[e] += yr[i]*nodeBel[y1+nStates*n2];
                    break;
                case 'P':
                    if (y1==y2)
                        g2[y1+nStates*e] -= yr[i]*2;
                    g2[y2+nStates*e] += yr[i]*nodeBel[y2+nStates*n1];
                    g2[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n2];
                    break;
                case 'S':
                    if ((y1+y2)%2)
                        g2[e] -= yr[i]*2;
                    g2[e] += yr[i]*nodeBel[(1+y2)%2+nStates*n1];
                    g2[e] += yr[i]*nodeBel[(1+y1)%2+nStates*n2];
                    break;
                case 'F':
                    g2[y1+nStates*(y2+nStates*e)] -= yr[i]*2;
                    for(s=0;s<nStates;s++) {
                        g2[s+nStates*(y2+nStates*e)] += yr[i]*nodeBel[s+nStates*n1];
                        g2[y1+nStates*(s+nStates*e)] += yr[i]*nodeBel[s+nStates*n2];
                    }
            }
        }
        
    }
    mxFree(logPot);
    mxFree(z);
    mxFree(nodeBel);
}
