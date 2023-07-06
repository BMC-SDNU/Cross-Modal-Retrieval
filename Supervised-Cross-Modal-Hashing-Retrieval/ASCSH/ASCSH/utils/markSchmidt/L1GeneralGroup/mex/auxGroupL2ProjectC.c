#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variable Declarations */
    int i, v, g, nVars, nGroups, *groupStart, *groupPtr;
    
    double *p, *x, groupNorm, alpha, avg;
    
    /* Input */
    x = mxGetPr(prhs[0]);
    groupStart = (int*)mxGetPr(prhs[1]);
    groupPtr = (int*)mxGetPr(prhs[2]);
    
    /* Compute Sizes */
    nGroups = mxGetDimensions(prhs[1])[0]-1;
    nVars = mxGetDimensions(prhs[0])[0]-nGroups;
    
    /*printf("nGroups = %d, nVars = %d\n",nGroups,nVars); */
    
    /* Output */
    plhs[0] = mxCreateNumericArray(2, mxGetDimensions(prhs[0]), mxDOUBLE_CLASS, mxREAL);
    p = mxGetPr(plhs[0]);
    
    for(v = 0; v < nVars; v++)
        p[v] = x[v];
    
    /* Do projection of each group */
    for(g = 0; g < nGroups; g++) {
        /* Compute norm of group */
        groupNorm = 0;
        for(v = groupStart[g]; v < groupStart[g+1]; v++) {
            groupNorm += x[groupPtr[v]]*x[groupPtr[v]];
        }
        groupNorm = sqrt(groupNorm);
        /*printf("Group Norm = %f\n",groupNorm);*/
        
        /* Compute alpha-I */
        alpha = x[nVars+g];
        /*printf("Alpha = %f\n",alpha); */
        
        if(alpha >= groupNorm) /* Bound is satisfied at (x,alpha) */ {
            for(v = groupStart[g]; v < groupStart[g+1]; v++) {
                p[groupPtr[v]] = x[groupPtr[v]];
            }
            p[nVars+g] = x[nVars+g];
        }
        else /* Bound is not satisfied, decrease x(group), increase (alpha) */ {
            avg = (groupNorm+alpha)/2;
            /*printf("Avg = %f\n",avg);*/
            
            if(avg < 0) {
                for(v = groupStart[g]; v < groupStart[g+1]; v++) {
                    p[groupPtr[v]] = 0;
                }
                p[nVars+g] = 0;
            }
            else {
                for(v = groupStart[g]; v < groupStart[g+1]; v++) {
                    p[groupPtr[v]] = x[groupPtr[v]]*avg/groupNorm;
                }
                p[nVars+g] = avg;
            }
        }
    }
    
}