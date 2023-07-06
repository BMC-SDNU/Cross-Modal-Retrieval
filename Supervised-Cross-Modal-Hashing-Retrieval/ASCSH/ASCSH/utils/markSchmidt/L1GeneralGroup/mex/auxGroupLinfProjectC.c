#include <math.h>
#include "mex.h"

void quickSort(double *x, int l, int r);
int  partition(double *x, int l, int r);

double absolute(double x) {
    if (x >= 0)
        return x;
    else
        return -x;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variable Declarations */
    int i, v, g, k, nVars, nGroups, *groupStart, *groupPtr, groupSize, satisfied;
    
    double *p, *x, groupNorm, alpha, avg, *w, *sorted, projPoint,s;
    
    /* Input */
    x = mxGetPr(prhs[0]);
    groupStart = (int*)mxGetPr(prhs[1]);
    groupPtr = (int*)mxGetPr(prhs[2]);
    
    /* Compute Sizes */
    nGroups = mxGetDimensions(prhs[1])[0]-1;
    nVars = mxGetDimensions(prhs[0])[0]-nGroups;
    
    w = mxCalloc(nVars,sizeof(double));
    sorted = mxCalloc(nVars+1,sizeof(double));
    
    /*printf("nGroups = %d, nVars = %d\n",nGroups,nVars); */
    
    /* Output */
    plhs[0] = mxCreateNumericArray(2, mxGetDimensions(prhs[0]), mxDOUBLE_CLASS, mxREAL);
    p = mxGetPr(plhs[0]);
    
    for(v = 0; v < nVars; v++)
        p[v] = x[v];
    
    /* Do projection of each group */
    for(g = 0; g < nGroups; g++) {
        groupSize = 0;
        for(v = groupStart[g]; v < groupStart[g+1]; v++)
            w[groupSize++] = x[groupPtr[v]];
        alpha = x[nVars+g];
        
        satisfied = 1;
        for(v = 0; v < groupSize; v++) {
            if (absolute(w[v]) > alpha) {
                satisfied = 0;
            }
        }
        
        if(satisfied) {
            for(v = groupStart[g]; v < groupStart[g+1]; v++) {
                p[groupPtr[v]] = x[groupPtr[v]];
            }
            p[nVars+g] = x[nVars+g];
            continue;
        }
        
        /*printf("Unsatisfied\n");*/
        
        for(v = 0; v < groupSize; v++) {
            sorted[v] = absolute(w[v]);
        }
        sorted[groupSize] = 0;
        quickSort(sorted,0,groupSize);
        
        /* for(v = 0; v <= groupSize; v++) {
            printf("%f ",sorted[v]);
         }
        printf("\n");
        */
        
        s = 0;
        for(k = 0; k <= groupSize; k++)
        {
            /* Compute Projection with k largest elements */
            s += sorted[k];
            projPoint = (s+alpha)/(k+2);
            
            if(projPoint > 0 && projPoint > sorted[k+1])
            {
                /* Optimal threshold is projPoint */
                for(v = groupStart[g]; v < groupStart[g+1]; v++)
                {
                    if(absolute(x[groupPtr[v]]) >= sorted[k])
                    {
                        if(x[groupPtr[v]] >= 0)
                        {
                            p[groupPtr[v]] = projPoint;
                        }
                        else
                        {
                            p[groupPtr[v]] = -projPoint;
                        }
                    }
                    else
                    {
                        p[groupPtr[v]] = x[groupPtr[v]];
                    }
                }
                p[nVars+g] = projPoint;
                break;
            }
            
            if(k == groupSize) {
                /* alpha is too negative, optimal answer is 0 */
                for(v = groupStart[g]; v < groupStart[g+1]; v++) {
                    p[groupPtr[v]] = 0;
                }
                p[nVars+g] = 0;
            }
        }
    }
    mxFree(w);
    mxFree(sorted);
}


/* ------------------------------------------------------------------ */
void quickSort(double *x, int l, int r)
/* ------------------------------------------------------------------ */
{  int j;

   if (l < r)
   {
      j = partition(x, l, r);
      quickSort(x, l,   j-1);
      quickSort(x, j+1, r  );
   }
}


/* ------------------------------------------------------------------ */
int  partition(double *x, int l, int r)
/* ------------------------------------------------------------------ */
{  double pivot, t;
   int    i, j;

   pivot = x[l];
   i     = l;
   j     = r+1;
		
   while(1)
   {
      do ++i; while(x[i] >= pivot && i <= r);
      do --j; while(x[j] <  pivot          );
      if (i >= j) break;

      /* Swap elements i and j */
      t    = x[i];
      x[i] = x[j];
      x[j] = t;
   }

   /* Swap elements l and j*/
   t    = x[l];
   x[l] = x[j];
   x[j] = t;

   return j;
}