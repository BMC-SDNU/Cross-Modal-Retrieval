#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Variable Declarations */
    int i,v,g,nVars,nGroups,nVarGroups,maxIter,*groupStart,*groupPtr;
    
    double *p,*x,*I,*x_old, groupNorm, alpha,avg,diff;
    
    /* Input */
    x = mxGetPr(prhs[0]);
    groupStart = (int*)mxGetPr(prhs[1]);
    groupPtr = (int*)mxGetPr(prhs[2]);
    
    /* Compute Sizes */
    nGroups = mxGetDimensions(prhs[1])[0]-1;
    nVars = mxGetDimensions(prhs[0])[0]-nGroups;
    nVarGroups = mxGetDimensions(prhs[2])[0];
    /* printf("nGroups = %d, nVars = %d, nVarGroups = %d\n",nGroups,nVars,nVarGroups); */
    
    /* Output */
    plhs[0] = mxCreateNumericArray(2,mxGetDimensions(prhs[0]),mxDOUBLE_CLASS,mxREAL);
    p = mxGetPr(plhs[0]);
    
    /* Initialize */
    maxIter = 1000;
    I = mxCalloc(nVarGroups+nGroups,sizeof(double));
    x_old = mxCalloc(nVars,sizeof(double));
    
    for(i = 0; i < maxIter; i++)
    {
        /* Set p to previous version of x */
        for(v = 0; v < nVars+nGroups;v++)
            p[v] = x[v];
        
        /* Do projection of each group */
        for(g = 0; g < nGroups; g++)
        {
            for(v = groupStart[g]; v < groupStart[g+1]; v++)
            {
                x_old[groupPtr[v]] = x[groupPtr[v]];
            }
            
            /* Compute norm of group at x-I */
            groupNorm = 0;
            for(v = groupStart[g]; v < groupStart[g+1]; v++)
            {
                groupNorm += (x[groupPtr[v]]-I[v])*(x[groupPtr[v]]-I[v]);
            }
            groupNorm = sqrt(groupNorm);
            /*printf("Group Norm = %f\n",groupNorm);*/
            
            /* Compute alpha-I */
            alpha = x[nVars+g]-I[nVarGroups+g];
            /*printf("Alpha = %f\n",alpha); */
            
            if(alpha >= groupNorm) /* Bound is satisfied at (x-I,alpha-I) */
            {
                for(v = groupStart[g]; v < groupStart[g+1]; v++)
                {
                    x[groupPtr[v]] = x[groupPtr[v]]-I[v];
                }
                x[nVars+g] = x[nVars+g]-I[nVarGroups+g];
            }
            else /* Bound is not satisfied, decrease x(group), increase (alpha) */
            {
                avg = (groupNorm+alpha)/2;
                /*printf("Avg = %f\n",avg);*/
                
                if(avg < 0)
                {
                    for(v = groupStart[g]; v < groupStart[g+1]; v++)
                    {
                        x[groupPtr[v]] = 0;
                    }
                    x[nVars+g] = 0;
                }
                else
                {
                    for(v = groupStart[g]; v < groupStart[g+1]; v++)
                    {
                        x[groupPtr[v]] = (x[groupPtr[v]]-I[v])*avg/groupNorm;
                    }
                    x[nVars+g] = avg;
                }
            }

            /* Update I */
            for(v = groupStart[g]; v < groupStart[g+1]; v++)
            {
                /*printf("Updating I[%d] = %f - (%f - %f)\n",v,x[groupPtr[v]],p[groupPtr[v]],I[v]);*/
                I[v] = x[groupPtr[v]] - (x_old[groupPtr[v]] - I[v]);
            }
            I[nVarGroups+g] = x[nVars+g] - (p[nVars+g] - I[nVarGroups+g]);
        }
        
        diff = 0;
        for(v = 0; v < nVars+nGroups; v++)
        {
            if(x[v]-p[v] > p[v]-x[v])
            {
                diff += x[v]-p[v];
            }
            else
            {
                diff += p[v]-x[v];
            }
        }
        /*printf("Iter = %d, Res = %.10f\n",i,diff);*/
        
        if(diff < 1e-10)
            break;
    
        /*
        for(v = 0; v < nVarGroups+g; v++)
        {
         printf("I[%d] = %f\n",v+1,I[v]);   
        }
         */
    }
    
    for(v = 0; v < nVars+nGroups;v++)
            p[v] = x[v];
    
    mxFree(I);
    mxFree(x_old);
}