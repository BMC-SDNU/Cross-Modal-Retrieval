#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* variables */
	char param;
    int i, s, n, e, n1, n2, n3, n4, n5, n6, n7, nInstances, nNodes, nStates, nEdges2, nEdges3, nEdges4, nEdges5, nEdges6, nEdges7,
            *edges2, *edges3, *edges4, *edges5, *edges6, *edges7, *y, y1, y2, y3, y4, y5, y6, y7;
    double *yr, *g1, *g2, *g3, *g4, *g5, *g6, *g7, *logPot, *z, *pseudoNLL, *w1, *w2, *w3, *w4, *w5, *w6, *w7, *nodeBel;
    
    /* input */
	param = *(mxChar*)mxGetData(prhs[0]);
    y = (int*)mxGetPr(prhs[1]);
    yr = mxGetPr(prhs[2]);
    g1 = mxGetPr(prhs[3]);
    g2 = mxGetPr(prhs[4]);
    g3 = mxGetPr(prhs[5]);
    g4 = mxGetPr(prhs[6]);
    g5 = mxGetPr(prhs[7]);
    g6 = mxGetPr(prhs[8]);
	g7 = mxGetPr(prhs[9]);
    edges2 = (int*)mxGetPr(prhs[10]);
    edges3 = (int*)mxGetPr(prhs[11]);
    edges4 = (int*)mxGetPr(prhs[12]);
    edges5 = (int*)mxGetPr(prhs[13]);
    edges6 = (int*)mxGetPr(prhs[14]);
	edges7 = (int*)mxGetPr(prhs[15]);
    w1 = mxGetPr(prhs[16]);
    w2 = mxGetPr(prhs[17]);
    w3 = mxGetPr(prhs[18]);
    w4 = mxGetPr(prhs[19]);
    w5 = mxGetPr(prhs[20]);
    w6 = mxGetPr(prhs[21]);
	w7 = mxGetPr(prhs[22]);
    
    /* compute sizes */
    nInstances = mxGetDimensions(prhs[1])[0];
    nNodes = mxGetDimensions(prhs[16])[0];
    nStates = mxGetDimensions(prhs[16])[1]+1;
    nEdges2 = mxGetDimensions(prhs[10])[0];
    nEdges3 = mxGetDimensions(prhs[11])[0];
    nEdges4 = mxGetDimensions(prhs[12])[0];
    nEdges5 = mxGetDimensions(prhs[13])[0];
    nEdges6 = mxGetDimensions(prhs[14])[0];
    nEdges7 = mxGetDimensions(prhs[15])[0];
    
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
        for(e = 0;e < nEdges2;e++) {
            n1 = edges2[e];
            n2 = edges2[e+nEdges2];
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
        for(e = 0;e < nEdges3;e++) {
            n1 = edges3[e];
            n2 = edges3[e+nEdges3];
            n3 = edges3[e+nEdges3+nEdges3];
            y1 = y[i + nInstances*n1];
            y2 = y[i + nInstances*n2];
            y3 = y[i + nInstances*n3];
            
            switch (param) {
				case 'C':
					if (y2==0 && y3==0)
						logPot[nStates*n1] += w3[e];
					if (y1==0 && y3==0)
						logPot[nStates*n2] += w3[e];
					if (y1==0 && y2==0)
						logPot[nStates*n3] += w3[e];
					break;
				case 'I':
					if (y2==y3)
						logPot[y2+nStates*n1] += w3[e];
					if (y1==y3)
						logPot[y1+nStates*n2] += w3[e];
					if (y1==y2)
						logPot[y1+nStates*n3] += w3[e];
					break;
				case 'P':
					if (y2==y3)
						logPot[y2+nStates*n1] += w3[y2+nStates*e];
					if (y1==y3)
						logPot[y1+nStates*n2] += w3[y1+nStates*e];
					if (y1==y2)
						logPot[y1+nStates*n3] += w3[y1+nStates*e];
					break;
				case 'S':
					logPot[((y2+y3) % 2) + nStates*n1] += w3[e];
					logPot[((y1+y3) % 2) + nStates*n2] += w3[e];
					logPot[((y1+y2) % 2) + nStates*n3] += w3[e];
					break;
				case 'F':
					for(s=0; s<nStates; s++) {
						logPot[s+nStates*n1] += w3[s+nStates*(y2+nStates*(y3+nStates*e))];
						logPot[s+nStates*n2] += w3[y1+nStates*(s+nStates*(y3+nStates*e))];
						logPot[s+nStates*n3] += w3[y1+nStates*(y2+nStates*(s+nStates*e))];
					}
			}
        }
		
        for(e = 0;e < nEdges4;e++) {
            n1 = edges4[e];
            n2 = edges4[e+nEdges4];
            n3 = edges4[e+nEdges4+nEdges4];
            n4 = edges4[e+nEdges4+nEdges4+nEdges4];
            y1 = y[i + nInstances*n1];
            y2 = y[i + nInstances*n2];
            y3 = y[i + nInstances*n3];
            y4 = y[i + nInstances*n4];
           
			switch (param) {
				case 'C':
					if (y2==0 && y3==0 && y4==0)
						logPot[nStates*n1] += w4[e];
					if (y1==0 && y3==0 && y4==0)
						logPot[nStates*n2] += w4[e];
					if (y1==0 && y2==0 && y4==0)
						logPot[nStates*n3] += w4[e];
					if (y1==0 && y2==0 && y3==0)
						logPot[nStates*n4] += w4[e];
					break;
				case 'I':
					if (y2==y3 && y3==y4)
						logPot[y2+nStates*n1] += w4[e];
					if (y1==y3 && y3==y4)
						logPot[y1+nStates*n2] += w4[e];
					if (y1==y2 && y2==y4)
						logPot[y1+nStates*n3] += w4[e];
					if (y1==y2 && y2==y3)
						logPot[y1+nStates*n4] += w4[e];
					break;
				case 'P':
					if (y2==y3 && y3==y4)
						logPot[y2+nStates*n1] += w4[y2+nStates*e];
					if (y1==y3 && y3==y4)
						logPot[y1+nStates*n2] += w4[y1+nStates*e];
					if (y1==y2 && y2==y4)
						logPot[y1+nStates*n3] += w4[y1+nStates*e];
					if (y1==y2 && y2==y3)
						logPot[y1+nStates*n4] += w4[y1+nStates*e];
					break;
				case 'S':
					logPot[((1+y2+y3+y4) % 2) + nStates*n1] += w4[e];
					logPot[((1+y1+y3+y4) % 2) + nStates*n2] += w4[e];
					logPot[((1+y1+y2+y4) % 2) + nStates*n3] += w4[e];
					logPot[((1+y1+y2+y3) % 2) + nStates*n4] += w4[e];
					break;
				case 'F':
					for(s=0; s<nStates; s++) {
						logPot[s+nStates*n1] += w4[s+nStates*(y2+nStates*(y3+nStates*(y4+nStates*e)))];
						logPot[s+nStates*n2] += w4[y1+nStates*(s+nStates*(y3+nStates*(y4+nStates*e)))];
						logPot[s+nStates*n3] += w4[y1+nStates*(y2+nStates*(s+nStates*(y4+nStates*e)))];
						logPot[s+nStates*n4] += w4[y1+nStates*(y2+nStates*(y3+nStates*(s+nStates*e)))];
					}
			}
        }
        for(e = 0;e < nEdges5;e++) {
            n1 = edges5[e];
            n2 = edges5[e+nEdges5];
            n3 = edges5[e+nEdges5+nEdges5];
            n4 = edges5[e+nEdges5+nEdges5+nEdges5];
            n5 = edges5[e+nEdges5+nEdges5+nEdges5+nEdges5];
            y1 = y[i + nInstances*n1];
            y2 = y[i + nInstances*n2];
            y3 = y[i + nInstances*n3];
            y4 = y[i + nInstances*n4];
            y5 = y[i + nInstances*n5];
			switch (param) {
				case 'C':
					if (y2==0 && y3==0 && y4==0 && y5==0)
						logPot[nStates*n1] += w5[e];
					if (y1==0 && y3==0 && y4==0 && y5==0)
						logPot[nStates*n2] += w5[e];
					if (y1==0 && y2==0 && y4==0 && y5==0)
						logPot[nStates*n3] += w5[e];
					if (y1==0 && y2==0 && y3==0 && y5==0)
						logPot[nStates*n4] += w5[e];
					if (y1==0 && y2==0 && y3==0 && y4==0)
						logPot[nStates*n5] += w5[e];
					break;
				case 'I':
					if (y2==y3 && y3==y4 && y4==y5)
						logPot[y2+nStates*n1] += w5[e];
					if (y1==y3 && y3==y4 && y4==y5)
						logPot[y1+nStates*n2] += w5[e];
					if (y1==y2 && y2==y4 && y4==y5)
						logPot[y1+nStates*n3] += w5[e];
					if (y1==y2 && y2==y3 && y3==y5)
						logPot[y1+nStates*n4] += w5[e];
					if (y1==y2 && y2==y3 && y3==y4)
						logPot[y1+nStates*n5] += w5[e];
					break;
				case 'P':
					if (y2==y3 && y3==y4 && y4==y5)
						logPot[y2+nStates*n1] += w5[y2+nStates*e];
					if (y1==y3 && y3==y4 && y4==y5)
						logPot[y1+nStates*n2] += w5[y1+nStates*e];
					if (y1==y2 && y2==y4 && y4==y5)
						logPot[y1+nStates*n3] += w5[y1+nStates*e];
					if (y1==y2 && y2==y3 && y3==y5)
						logPot[y1+nStates*n4] += w5[y1+nStates*e];
					if (y1==y2 && y2==y3 && y3==y4)
						logPot[y1+nStates*n5] += w5[y1+nStates*e];
					break;
				case 'S':
					logPot[((y2+y3+y4+y5) % 2) + nStates*n1] += w5[e];
					logPot[((y1+y3+y4+y5) % 2) + nStates*n2] += w5[e];
					logPot[((y1+y2+y4+y5) % 2) + nStates*n3] += w5[e];
					logPot[((y1+y2+y3+y5) % 2) + nStates*n4] += w5[e];
					logPot[((y1+y2+y3+y4) % 2) + nStates*n5] += w5[e];
					break;
				case 'F':
					for(s=0; s<nStates; s++) {
						logPot[s+nStates*n1] += w5[s+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*e))))];
						logPot[s+nStates*n2] += w5[y1+nStates*(s+nStates*(y3+nStates*(y4+nStates*(y5+nStates*e))))];
						logPot[s+nStates*n3] += w5[y1+nStates*(y2+nStates*(s+nStates*(y4+nStates*(y5+nStates*e))))];
						logPot[s+nStates*n4] += w5[y1+nStates*(y2+nStates*(y3+nStates*(s+nStates*(y5+nStates*e))))];
						logPot[s+nStates*n5] += w5[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(s+nStates*e))))];
					}
        }
		}
        for(e = 0;e < nEdges6;e++) {
            n1 = edges6[e];
            n2 = edges6[e+nEdges6];
            n3 = edges6[e+nEdges6*2];
            n4 = edges6[e+nEdges6*3];
            n5 = edges6[e+nEdges6*4];
            n6 = edges6[e+nEdges6*5];
            y1 = y[i + nInstances*n1];
            y2 = y[i + nInstances*n2];
            y3 = y[i + nInstances*n3];
            y4 = y[i + nInstances*n4];
            y5 = y[i + nInstances*n5];
            y6 = y[i + nInstances*n6];
			switch (param) {
				case 'C':
					if (y2==0 && y3==0 && y4==0 && y5==0 && y6==0)
						logPot[nStates*n1] += w6[e];
					if (y1==0 && y3==0 && y4==0 && y5==0 && y6==0)
						logPot[nStates*n2] += w6[e];
					if (y1==0 && y2==0 && y4==0 && y5==0 && y6==0)
						logPot[nStates*n3] += w6[e];
					if (y1==0 && y2==0 && y3==0 && y5==0 && y6==0)
						logPot[nStates*n4] += w6[e];
					if (y1==0 && y2==0 && y3==0 && y4==0 && y6==0)
						logPot[nStates*n5] += w6[e];
					if (y1==0 && y2==0 && y3==0 && y4==0 && y5==0)
						logPot[nStates*n6] += w6[e];
					break;
				case 'I':
					if (y2==y3 && y3==y4 && y4==y5 && y5==y6)
						logPot[y2+nStates*n1] += w6[e];
					if (y1==y3 && y3==y4 && y4==y5 && y5==y6)
						logPot[y1+nStates*n2] += w6[e];
					if (y1==y2 && y2==y4 && y4==y5 && y5==y6)
						logPot[y1+nStates*n3] += w6[e];
					if (y1==y2 && y2==y3 && y3==y5 && y5==y6)
						logPot[y1+nStates*n4] += w6[e];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y6)
						logPot[y1+nStates*n5] += w6[e];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5)
						logPot[y1+nStates*n6] += w6[e];
					break;
				case 'P':
					if (y2==y3 && y3==y4 && y4==y5 && y5==y6)
						logPot[y2+nStates*n1] += w6[y2+nStates*e];
					if (y1==y3 && y3==y4 && y4==y5 && y5==y6)
						logPot[y1+nStates*n2] += w6[y1+nStates*e];
					if (y1==y2 && y2==y4 && y4==y5 && y5==y6)
						logPot[y1+nStates*n3] += w6[y1+nStates*e];
					if (y1==y2 && y2==y3 && y3==y5 && y5==y6)
						logPot[y1+nStates*n4] += w6[y1+nStates*e];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y6)
						logPot[y1+nStates*n5] += w6[y1+nStates*e];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5)
						logPot[y1+nStates*n6] += w6[y1+nStates*e];
					break;
				case 'S':
					logPot[((1+y2+y3+y4+y5+y6) % 2) + nStates*n1] += w6[e];
					logPot[((1+y1+y3+y4+y5+y6) % 2) + nStates*n2] += w6[e];
					logPot[((1+y1+y2+y4+y5+y6) % 2) + nStates*n3] += w6[e];
					logPot[((1+y1+y2+y3+y5+y6) % 2) + nStates*n4] += w6[e];
					logPot[((1+y1+y2+y3+y4+y6) % 2) + nStates*n5] += w6[e];
					logPot[((1+y1+y2+y3+y4+y5) % 2) + nStates*n6] += w6[e];
					break;
				case 'F':
					for(s=0; s<nStates; s++) {
						logPot[s+nStates*n1] += w6[s+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(y6+nStates*e)))))];
						logPot[s+nStates*n2] += w6[y1+nStates*(s+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(y6+nStates*e)))))];
						logPot[s+nStates*n3] += w6[y1+nStates*(y2+nStates*(s+nStates*(y4+nStates*(y5+nStates*(y6+nStates*e)))))];
						logPot[s+nStates*n4] += w6[y1+nStates*(y2+nStates*(y3+nStates*(s+nStates*(y5+nStates*(y6+nStates*e)))))];
						logPot[s+nStates*n5] += w6[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(s+nStates*(y6+nStates*e)))))];
						logPot[s+nStates*n6] += w6[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(s+nStates*e)))))];
					}
        }
        }
			for(e = 0;e < nEdges7;e++) {
            n1 = edges7[e];
            n2 = edges7[e+nEdges7];
            n3 = edges7[e+nEdges7*2];
            n4 = edges7[e+nEdges7*3];
            n5 = edges7[e+nEdges7*4];
            n6 = edges7[e+nEdges7*5];
            n7 = edges7[e+nEdges7*6];
            y1 = y[i + nInstances*n1];
            y2 = y[i + nInstances*n2];
            y3 = y[i + nInstances*n3];
            y4 = y[i + nInstances*n4];
            y5 = y[i + nInstances*n5];
            y6 = y[i + nInstances*n6];
            y7 = y[i + nInstances*n7];
			switch (param) {
				case 'C':
					if (y2==0 && y3==0 && y4==0 && y5==0 && y6==0 && y7==0)
						logPot[nStates*n1] += w7[e];
					if (y1==0 && y3==0 && y4==0 && y5==0 && y6==0 && y7==0)
						logPot[nStates*n2] += w7[e];
					if (y1==0 && y2==0 && y4==0 && y5==0 && y6==0 && y7==0)
						logPot[nStates*n3] += w7[e];
					if (y1==0 && y2==0 && y3==0 && y5==0 && y6==0 && y7==0)
						logPot[nStates*n4] += w7[e];
					if (y1==0 && y2==0 && y3==0 && y4==0 && y6==0 && y7==0)
						logPot[nStates*n5] += w7[e];
					if (y1==0 && y2==0 && y3==0 && y4==0 && y5==0 && y7==0)
						logPot[nStates*n6] += w7[e];
					if (y1==0 && y2==0 && y3==0 && y4==0 && y5==0 && y6==0)
						logPot[nStates*n7] += w7[e];
					break;
				case 'I':
					if (y2==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7)
						logPot[y2+nStates*n1] += w7[e];
					if (y1==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7)
						logPot[y1+nStates*n2] += w7[e];
					if (y1==y2 && y2==y4 && y4==y5 && y5==y6 && y6==y7)
						logPot[y1+nStates*n3] += w7[e];
					if (y1==y2 && y2==y3 && y3==y5 && y5==y6 && y6==y7)
						logPot[y1+nStates*n4] += w7[e];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y6 && y6==y7)
						logPot[y1+nStates*n5] += w7[e];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y7)
						logPot[y1+nStates*n6] += w7[e];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6)
						logPot[y1+nStates*n7] += w7[e];
					break;
				case 'P':
					if (y2==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7)
						logPot[y2+nStates*n1] += w7[y2+nStates*e];
					if (y1==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7)
						logPot[y1+nStates*n2] += w7[y1+nStates*e];
					if (y1==y2 && y2==y4 && y4==y5 && y5==y6 && y6==y7)
						logPot[y1+nStates*n3] += w7[y1+nStates*e];
					if (y1==y2 && y2==y3 && y3==y5 && y5==y6 && y6==y7)
						logPot[y1+nStates*n4] += w7[y1+nStates*e];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y6 && y6==y7)
						logPot[y1+nStates*n5] += w7[y1+nStates*e];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y7)
						logPot[y1+nStates*n6] += w7[y1+nStates*e];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6)
						logPot[y1+nStates*n7] += w7[y1+nStates*e];
					break;
				case 'S':
					logPot[((y2+y3+y4+y5+y6+y7) % 2) + nStates*n1] += w7[e];
					logPot[((y1+y3+y4+y5+y6+y7) % 2) + nStates*n2] += w7[e];
					logPot[((y1+y2+y4+y5+y6+y7) % 2) + nStates*n3] += w7[e];
					logPot[((y1+y2+y3+y5+y6+y7) % 2) + nStates*n4] += w7[e];
					logPot[((y1+y2+y3+y4+y6+y7) % 2) + nStates*n5] += w7[e];
					logPot[((y1+y2+y3+y4+y5+y7) % 2) + nStates*n6] += w7[e];
					logPot[((y1+y2+y3+y4+y5+y6) % 2) + nStates*n7] += w7[e];
					break;
				case 'F':
					for(s=0; s<nStates; s++) {
						logPot[s+nStates*n1] += w7[s+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(y6+nStates*(y7+nStates*e))))))];
						logPot[s+nStates*n2] += w7[y1+nStates*(s+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(y6+nStates*(y7+nStates*e))))))];
						logPot[s+nStates*n3] += w7[y1+nStates*(y2+nStates*(s+nStates*(y4+nStates*(y5+nStates*(y6+nStates*(y7+nStates*e))))))];
						logPot[s+nStates*n4] += w7[y1+nStates*(y2+nStates*(y3+nStates*(s+nStates*(y5+nStates*(y6+nStates*(y7+nStates*e))))))];
						logPot[s+nStates*n5] += w7[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(s+nStates*(y6+nStates*(y7+nStates*e))))))];
						logPot[s+nStates*n6] += w7[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(s+nStates*(y7+nStates*e))))))];
						logPot[s+nStates*n7] += w7[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(y6+nStates*(s+nStates*e))))))];
					}
        }
        }
		
		/*for(s = 0; s < nStates; s++) {
			printf("logPot(%d,:) = [", s);
			for(n = 0;n < nNodes; n++) {
				printf(" %f", logPot[s+nStates*n]);
			}
			printf(" ]\n");
		}*/

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
            for(s = 0; s < nStates; s++) {
				printf("nodeBel(%d,%d) = %f\n",s+1,n+1,nodeBel[s+nStates*n]);
            }
        }*/
        
        for(n = 0;n < nNodes;n++) {
            y1 = y[i + nInstances*n];
            
            if(y1 < nStates-1)
                g1[n + nNodes*y1] -= yr[i]*1;
            
			for(s=0; s < nStates-1; s++)
				g1[n + nNodes*s] += yr[i]*nodeBel[s+nStates*n];
        }
        for(e = 0;e < nEdges2;e++) {
            n1 = edges2[e];
            n2 = edges2[e+nEdges2];
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
        for(e = 0;e < nEdges3;e++) {
            n1 = edges3[e];
            n2 = edges3[e+nEdges3];
            n3 = edges3[e+nEdges3+nEdges3];
            y1 = y[i + nInstances*n1];
            y2 = y[i + nInstances*n2];
            y3 = y[i + nInstances*n3];
			
			switch (param) {
				case 'C':
					if (y1==0 && y2==0 && y3==0)
						g3[e] -= yr[i]*3;
					if (y2==0 && y3==0)
						g3[e] += yr[i]*nodeBel[nStates*n1];
					if (y1==0 && y3==0)
						g3[e] += yr[i]*nodeBel[nStates*n2];
					if (y1==0 && y2==0)
						g3[e] += yr[i]*nodeBel[nStates*n3];
					break;
				case 'I':
					if (y1==y2 && y2==y3)
						g3[e] -= yr[i]*3;
					if (y2==y3)
					g3[e] += yr[i]*nodeBel[y2+nStates*n1];
					if (y1==y3)
					g3[e] += yr[i]*nodeBel[y1+nStates*n2];
					if (y1==y2)
					g3[e] += yr[i]*nodeBel[y1+nStates*n3];
					break;
				case 'P':
					if (y1==y2 && y2==y3)
						g3[y1+nStates*e] -= yr[i]*3;
					if (y2==y3)
						g3[y2+nStates*e] += yr[i]*nodeBel[y2+nStates*n1];
					if (y1==y3)
						g3[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n2];
					if (y1==y2)
						g3[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n3];
					break;
				case 'S':
					if ((1+y1+y2+y3)%2)
						g3[e] -= yr[i]*3;
					g3[e] += yr[i]*nodeBel[(y2+y3)%2+nStates*n1];
					g3[e] += yr[i]*nodeBel[(y1+y3)%2+nStates*n2];
					g3[e] += yr[i]*nodeBel[(y1+y2)%2+nStates*n3];
					break;
				case 'F':
					g3[y1+nStates*(y2+nStates*(y3+nStates*e))] -= yr[i]*3;
					for(s=0;s<nStates;s++) {
						g3[s+nStates*(y2+nStates*(y3+nStates*e))] += yr[i]*nodeBel[s+nStates*n1];
						g3[y1+nStates*(s+nStates*(y3+nStates*e))] += yr[i]*nodeBel[s+nStates*n2];
						g3[y1+nStates*(y2+nStates*(s+nStates*e))] += yr[i]*nodeBel[s+nStates*n3];
					}
			}
        }
        for(e = 0;e < nEdges4;e++) {
            n1 = edges4[e];
            n2 = edges4[e+nEdges4];
            n3 = edges4[e+nEdges4+nEdges4];
            n4 = edges4[e+nEdges4+nEdges4+nEdges4];
            y1 = y[i + nInstances*n1];
            y2 = y[i + nInstances*n2];
            y3 = y[i + nInstances*n3];
            y4 = y[i + nInstances*n4];
			switch (param) {
				case 'C':
					if (y1==0 && y2==0 && y3==0 && y4==0)
						g4[e] -= yr[i]*4;
					if (y2==0 && y3==0 && y4==0)
						g4[e] += yr[i]*nodeBel[nStates*n1];
					if (y1==0 && y3==0 && y4==0)
						g4[e] += yr[i]*nodeBel[nStates*n2];
					if (y1==0 && y2==0 && y4==0)
						g4[e] += yr[i]*nodeBel[nStates*n3];
					if (y1==0 && y2==0 && y3==0)
						g4[e] += yr[i]*nodeBel[nStates*n4];
					break;
				case 'I':
					if (y1==y2 && y2==y3 && y3==y4)
						g4[e] -= yr[i]*4;
					if (y2==y3 && y3==y4)
					g4[e] += yr[i]*nodeBel[y2+nStates*n1];
					if (y1==y3 && y3==y4)
					g4[e] += yr[i]*nodeBel[y1+nStates*n2];
					if (y1==y2 && y2==y4)
					g4[e] += yr[i]*nodeBel[y1+nStates*n3];
					if (y1==y2 && y2==y3)
					g4[e] += yr[i]*nodeBel[y1+nStates*n4];
					break;
				case 'P':
					if (y1==y2 && y2==y3 && y3==y4)
						g4[y1+nStates*e] -= yr[i]*4;
					if (y2==y3 && y3==y4)
						g4[y2+nStates*e] += yr[i]*nodeBel[y2+nStates*n1];
					if (y1==y3 && y3==y4)
						g4[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n2];
					if (y1==y2 && y2==y4)
						g4[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n3];
					if (y1==y2 && y2==y3)
						g4[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n4];
					break;
				case 'S':
					if ((y1+y2+y3+y4)%2)
						g4[e] -= yr[i]*4;
					g4[e] += yr[i]*nodeBel[(1+y2+y3+y4)%2+nStates*n1];
					g4[e] += yr[i]*nodeBel[(1+y1+y3+y4)%2+nStates*n2];
					g4[e] += yr[i]*nodeBel[(1+y1+y2+y4)%2+nStates*n3];
					g4[e] += yr[i]*nodeBel[(1+y1+y2+y3)%2+nStates*n4];
					break;
				case 'F':
					g4[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*e)))] -= yr[i]*4;
					for(s=0;s<nStates;s++) {
						g4[s+nStates*(y2+nStates*(y3+nStates*(y4+nStates*e)))] += yr[i]*nodeBel[s+nStates*n1];
						g4[y1+nStates*(s+nStates*(y3+nStates*(y4+nStates*e)))] += yr[i]*nodeBel[s+nStates*n2];
						g4[y1+nStates*(y2+nStates*(s+nStates*(y4+nStates*e)))] += yr[i]*nodeBel[s+nStates*n3];
						g4[y1+nStates*(y2+nStates*(y3+nStates*(s+nStates*e)))] += yr[i]*nodeBel[s+nStates*n4];
					}
			}
        }
        for(e = 0;e < nEdges5;e++) {
            n1 = edges5[e];
            n2 = edges5[e+nEdges5];
            n3 = edges5[e+nEdges5+nEdges5];
            n4 = edges5[e+nEdges5+nEdges5+nEdges5];
            n5 = edges5[e+nEdges5+nEdges5+nEdges5+nEdges5];
            y1 = y[i + nInstances*n1];
            y2 = y[i + nInstances*n2];
            y3 = y[i + nInstances*n3];
            y4 = y[i + nInstances*n4];
            y5 = y[i + nInstances*n5];
            switch (param) {
				case 'C':
					if (y1==0 && y2==0 && y3==0 && y4==0 && y5==0)
						g5[e] -= yr[i]*5;
					if (y2==0 && y3==0 && y4==0 && y5==0)
						g5[e] += yr[i]*nodeBel[nStates*n1];
					if (y1==0 && y3==0 && y4==0 && y5==0)
						g5[e] += yr[i]*nodeBel[nStates*n2];
					if (y1==0 && y2==0 && y4==0 && y5==0)
						g5[e] += yr[i]*nodeBel[nStates*n3];
					if (y1==0 && y2==0 && y3==0 && y5==0)
						g5[e] += yr[i]*nodeBel[nStates*n4];
					if (y1==0 && y2==0 && y3==0 && y4==0)
						g5[e] += yr[i]*nodeBel[nStates*n5];
					break;
				case 'I':
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5)
						g5[e] -= yr[i]*5;
					if (y2==y3 && y3==y4 && y4==y5)
					g5[e] += yr[i]*nodeBel[y2+nStates*n1];
					if (y1==y3 && y3==y4 && y4==y5)
					g5[e] += yr[i]*nodeBel[y1+nStates*n2];
					if (y1==y2 && y2==y4 && y4==y5)
					g5[e] += yr[i]*nodeBel[y1+nStates*n3];
					if (y1==y2 && y2==y3 && y3==y5)
					g5[e] += yr[i]*nodeBel[y1+nStates*n4];
					if (y1==y2 && y2==y3 && y3==y4)
					g5[e] += yr[i]*nodeBel[y1+nStates*n5];
					break;
				case 'P':
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5)
						g5[y1+nStates*e] -= yr[i]*5;
					if (y2==y3 && y3==y4 && y4==y5)
						g5[y2+nStates*e] += yr[i]*nodeBel[y2+nStates*n1];
					if (y1==y3 && y3==y4 && y4==y5)
						g5[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n2];
					if (y1==y2 && y2==y4 && y4==y5)
						g5[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n3];
					if (y1==y2 && y2==y3 && y3==y5)
						g5[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n4];
					if (y1==y2 && y2==y3 && y3==y4)
						g5[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n5];
					break;
				case 'S':
					if ((1+y1+y2+y3+y4+y5)%2)
						g5[e] -= yr[i]*5;
					g5[e] += yr[i]*nodeBel[(y2+y3+y4+y5)%2+nStates*n1];
					g5[e] += yr[i]*nodeBel[(y1+y3+y4+y5)%2+nStates*n2];
					g5[e] += yr[i]*nodeBel[(y1+y2+y4+y5)%2+nStates*n3];
					g5[e] += yr[i]*nodeBel[(y1+y2+y3+y5)%2+nStates*n4];
					g5[e] += yr[i]*nodeBel[(y1+y2+y3+y4)%2+nStates*n5];
					break;
				case 'F':
					g5[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*e))))] -= yr[i]*5;
					for(s=0;s<nStates;s++) {
						g5[s+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*e))))] += yr[i]*nodeBel[s+nStates*n1];
						g5[y1+nStates*(s+nStates*(y3+nStates*(y4+nStates*(y5+nStates*e))))] += yr[i]*nodeBel[s+nStates*n2];
						g5[y1+nStates*(y2+nStates*(s+nStates*(y4+nStates*(y5+nStates*e))))] += yr[i]*nodeBel[s+nStates*n3];
						g5[y1+nStates*(y2+nStates*(y3+nStates*(s+nStates*(y5+nStates*e))))] += yr[i]*nodeBel[s+nStates*n4];
						g5[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(s+nStates*e))))] += yr[i]*nodeBel[s+nStates*n5];
					}
			}
        }
        for(e = 0;e < nEdges6;e++) {
            n1 = edges6[e];
            n2 = edges6[e+nEdges6];
            n3 = edges6[e+nEdges6*2];
            n4 = edges6[e+nEdges6*3];
            n5 = edges6[e+nEdges6*4];
            n6 = edges6[e+nEdges6*5];
            y1 = y[i + nInstances*n1];
            y2 = y[i + nInstances*n2];
            y3 = y[i + nInstances*n3];
            y4 = y[i + nInstances*n4];
            y5 = y[i + nInstances*n5];
            y6 = y[i + nInstances*n6];
			switch (param) {
				case 'C':
					if (y1==0 && y2==0 && y3==0 && y4==0 && y5==0 && y6==0)
						g6[e] -= yr[i]*6;
					if (y2==0 && y3==0 && y4==0 && y5==0 && y6==0)
						g6[e] += yr[i]*nodeBel[nStates*n1];
					if (y1==0 && y3==0 && y4==0 && y5==0 && y6==0)
						g6[e] += yr[i]*nodeBel[nStates*n2];
					if (y1==0 && y2==0 && y4==0 && y5==0 && y6==0)
						g6[e] += yr[i]*nodeBel[nStates*n3];
					if (y1==0 && y2==0 && y3==0 && y5==0 && y6==0)
						g6[e] += yr[i]*nodeBel[nStates*n4];
					if (y1==0 && y2==0 && y3==0 && y4==0 && y6==0)
						g6[e] += yr[i]*nodeBel[nStates*n5];
					if (y1==0 && y2==0 && y3==0 && y4==0 && y5==0)
						g6[e] += yr[i]*nodeBel[nStates*n6];
					break;
				case 'I':
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6)
						g6[e] -= yr[i]*6;
					if (y2==y3 && y3==y4 && y4==y5 && y5==y6)
					g6[e] += yr[i]*nodeBel[y2+nStates*n1];
					if (y1==y3 && y3==y4 && y4==y5 && y5==y6)
					g6[e] += yr[i]*nodeBel[y1+nStates*n2];
					if (y1==y2 && y2==y4 && y4==y5 && y5==y6)
					g6[e] += yr[i]*nodeBel[y1+nStates*n3];
					if (y1==y2 && y2==y3 && y3==y5 && y5==y6)
					g6[e] += yr[i]*nodeBel[y1+nStates*n4];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y6)
					g6[e] += yr[i]*nodeBel[y1+nStates*n5];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5)
					g6[e] += yr[i]*nodeBel[y1+nStates*n6];
					break;
				case 'P':
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6)
						g6[y1+nStates*e] -= yr[i]*6;
					if (y2==y3 && y3==y4 && y4==y5 && y5==y6)
						g6[y2+nStates*e] += yr[i]*nodeBel[y2+nStates*n1];
					if (y1==y3 && y3==y4 && y4==y5 && y5==y6)
						g6[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n2];
					if (y1==y2 && y2==y4 && y4==y5 && y5==y6)
						g6[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n3];
					if (y1==y2 && y2==y3 && y3==y5 && y5==y6)
						g6[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n4];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y6)
						g6[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n5];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5)
						g6[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n6];
					break;
				case 'S':
					if ((y1+y2+y3+y4+y5+y6)%2)
						g6[e] -= yr[i]*6;
					g6[e] += yr[i]*nodeBel[(1+y2+y3+y4+y5+y6)%2+nStates*n1];
					g6[e] += yr[i]*nodeBel[(1+y1+y3+y4+y5+y6)%2+nStates*n2];
					g6[e] += yr[i]*nodeBel[(1+y1+y2+y4+y5+y6)%2+nStates*n3];
					g6[e] += yr[i]*nodeBel[(1+y1+y2+y3+y5+y6)%2+nStates*n4];
					g6[e] += yr[i]*nodeBel[(1+y1+y2+y3+y4+y6)%2+nStates*n5];
					g6[e] += yr[i]*nodeBel[(1+y1+y2+y3+y4+y5)%2+nStates*n6];
					break;
				case 'F':
					g6[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(y6+nStates*e)))))] -= yr[i]*6;
					for(s=0;s<nStates;s++) {
						g6[s+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(y6+nStates*e)))))] += yr[i]*nodeBel[s+nStates*n1];
						g6[y1+nStates*(s+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(y6+nStates*e)))))] += yr[i]*nodeBel[s+nStates*n2];
						g6[y1+nStates*(y2+nStates*(s+nStates*(y4+nStates*(y5+nStates*(y6+nStates*e)))))] += yr[i]*nodeBel[s+nStates*n3];
						g6[y1+nStates*(y2+nStates*(y3+nStates*(s+nStates*(y5+nStates*(y6+nStates*e)))))] += yr[i]*nodeBel[s+nStates*n4];
						g6[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(s+nStates*(y6+nStates*e)))))] += yr[i]*nodeBel[s+nStates*n5];
						g6[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(s+nStates*e)))))] += yr[i]*nodeBel[s+nStates*n6];
					}
			}
        }
		for(e = 0;e < nEdges7;e++) {
            n1 = edges7[e];
            n2 = edges7[e+nEdges7];
            n3 = edges7[e+nEdges7*2];
            n4 = edges7[e+nEdges7*3];
            n5 = edges7[e+nEdges7*4];
            n6 = edges7[e+nEdges7*5];
            n7 = edges7[e+nEdges7*6];
            y1 = y[i + nInstances*n1];
            y2 = y[i + nInstances*n2];
            y3 = y[i + nInstances*n3];
            y4 = y[i + nInstances*n4];
            y5 = y[i + nInstances*n5];
            y6 = y[i + nInstances*n6];
            y7 = y[i + nInstances*n7];
			switch (param) {
				case 'C':
					if (y1==0 && y2==0 && y3==0 && y4==0 && y5==0 && y6==0 && y7==0)
						g7[e] -= yr[i]*7;
					if (y2==0 && y3==0 && y4==0 && y5==0 && y6==0 && y7==0)
						g7[e] += yr[i]*nodeBel[nStates*n1];
					if (y1==0 && y3==0 && y4==0 && y5==0 && y6==0 && y7==0)
						g7[e] += yr[i]*nodeBel[nStates*n2];
					if (y1==0 && y2==0 && y4==0 && y5==0 && y6==0 && y7==0)
						g7[e] += yr[i]*nodeBel[nStates*n3];
					if (y1==0 && y2==0 && y3==0 && y5==0 && y6==0 && y7==0)
						g7[e] += yr[i]*nodeBel[nStates*n4];
					if (y1==0 && y2==0 && y3==0 && y4==0 && y6==0 && y7==0)
						g7[e] += yr[i]*nodeBel[nStates*n5];
					if (y1==0 && y2==0 && y3==0 && y4==0 && y5==0 && y7==0)
						g7[e] += yr[i]*nodeBel[nStates*n6];
					if (y1==0 && y2==0 && y3==0 && y4==0 && y5==0 && y6==0)
						g7[e] += yr[i]*nodeBel[nStates*n7];
					break;
				case 'I':
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7)
						g7[e] -= yr[i]*7;
					if (y2==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7)
					g7[e] += yr[i]*nodeBel[y2+nStates*n1];
					if (y1==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7)
					g7[e] += yr[i]*nodeBel[y1+nStates*n2];
					if (y1==y2 && y2==y4 && y4==y5 && y5==y6 && y6==y7)
					g7[e] += yr[i]*nodeBel[y1+nStates*n3];
					if (y1==y2 && y2==y3 && y3==y5 && y5==y6 && y6==y7)
					g7[e] += yr[i]*nodeBel[y1+nStates*n4];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y6 && y6==y7)
					g7[e] += yr[i]*nodeBel[y1+nStates*n5];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y7)
					g7[e] += yr[i]*nodeBel[y1+nStates*n6];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6)
					g7[e] += yr[i]*nodeBel[y1+nStates*n7];
					break;
				case 'P':
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7)
						g7[y1+nStates*e] -= yr[i]*7;
					if (y2==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7)
						g7[y2+nStates*e] += yr[i]*nodeBel[y2+nStates*n1];
					if (y1==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7)
						g7[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n2];
					if (y1==y2 && y2==y4 && y4==y5 && y5==y6 && y6==y7)
						g7[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n3];
					if (y1==y2 && y2==y3 && y3==y5 && y5==y6 && y6==y7)
						g7[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n4];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y6 && y6==y7)
						g7[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n5];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y7)
						g7[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n6];
					if (y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6)
						g7[y1+nStates*e] += yr[i]*nodeBel[y1+nStates*n7];
					break;
				case 'S':
					if ((1+y1+y2+y3+y4+y5+y6+y7)%2)
						g7[e] -= yr[i]*7;
					g7[e] += yr[i]*nodeBel[(y2+y3+y4+y5+y6+y7)%2+nStates*n1];
					g7[e] += yr[i]*nodeBel[(y1+y3+y4+y5+y6+y7)%2+nStates*n2];
					g7[e] += yr[i]*nodeBel[(y1+y2+y4+y5+y6+y7)%2+nStates*n3];
					g7[e] += yr[i]*nodeBel[(y1+y2+y3+y5+y6+y7)%2+nStates*n4];
					g7[e] += yr[i]*nodeBel[(y1+y2+y3+y4+y6+y7)%2+nStates*n5];
					g7[e] += yr[i]*nodeBel[(y1+y2+y3+y4+y5+y7)%2+nStates*n6];
					g7[e] += yr[i]*nodeBel[(y1+y2+y3+y4+y5+y6)%2+nStates*n7];
					break;
				case 'F':
					g7[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(y6+nStates*(y7+nStates*e))))))] -= yr[i]*7;
					for(s=0;s<nStates;s++) {
						g7[s+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(y6+nStates*(y7+nStates*e))))))] += yr[i]*nodeBel[s+nStates*n1];
						g7[y1+nStates*(s+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(y6+nStates*(y7+nStates*e))))))] += yr[i]*nodeBel[s+nStates*n2];
						g7[y1+nStates*(y2+nStates*(s+nStates*(y4+nStates*(y5+nStates*(y6+nStates*(y7+nStates*e))))))] += yr[i]*nodeBel[s+nStates*n3];
						g7[y1+nStates*(y2+nStates*(y3+nStates*(s+nStates*(y5+nStates*(y6+nStates*(y7+nStates*e))))))] += yr[i]*nodeBel[s+nStates*n4];
						g7[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(s+nStates*(y6+nStates*(y7+nStates*e))))))] += yr[i]*nodeBel[s+nStates*n5];
						g7[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(s+nStates*(y7+nStates*e))))))] += yr[i]*nodeBel[s+nStates*n6];
						g7[y1+nStates*(y2+nStates*(y3+nStates*(y4+nStates*(y5+nStates*(y6+nStates*(s+nStates*e))))))] += yr[i]*nodeBel[s+nStates*n7];
					}
			}
        }
    }
    mxFree(logPot);
    mxFree(z);
    mxFree(nodeBel);
}
