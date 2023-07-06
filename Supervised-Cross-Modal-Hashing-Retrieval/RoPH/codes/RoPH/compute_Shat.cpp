#include "mex.h"
#include "matrix.h"
#include <omp.h>
#include <inttypes.h>
#include <cmath>
// #include <stdlib.h>

#define DEBUG 0

const int lookup [] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};

//using namespace std;

//to compile, run:
//mex compute_Shat.cpp -largeArrayDims  COMPFLAGS="/openmp $COMPFLAGS"
//to compile in Linux, run:
//mex -O -g compute_Shat.cpp -largeArrayDims CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"

inline int match(uint8_t*P, uint8_t*Q, int codelb)
{
	int output = 0;
    for (int i=0; i<codelb; i++) 
        output+= lookup[P[i] ^ Q[i]];
    return output;
}

void deal_one_sample(uint8_t*Bcb, uint8_t* b, uint32_t*ind_pos, uint32_t*ind_neg, uint32_t N, uint32_t T, uint32_t K, int* shat)
{
int *dist = new int[N];
#pragma omp parallel for
for(int i=0;i<N;i++){
    dist[i] = match(b, Bcb+i*K, K);
}

#pragma omp parallel for
for(int i=0;i<T;i++){
    shat[i] = 2*(dist[ind_neg[i]] - dist[ind_pos[i]]);
}

delete[] dist;
}

//mex function
void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
uint32_t*triplets;
uint8_t* Bcb;
uint32_t K,T,N;
int i, j;

if(nrhs!=2){
    mexErrMsgTxt("There are 2 inputs!\n");
    mexPrintf("usage:\tShat=compute_Shat(triplets,Bcb),where triplets are 0-based index.\n");
}

if(!mxIsUint32(prhs[0])) mexErrMsgTxt("[1]: triplets must be uint32");
if(!mxIsUint8(prhs[1])) mexErrMsgTxt("[1]: Bcb must be uint8");

//get inputs
K = mxGetM(prhs[1]);//number of bytes
N = mxGetN(prhs[1]);
triplets = (uint32_t*)mxGetPr(prhs[0]);
T = mxGetM(prhs[0])/N;
if(floor((mxGetM(prhs[0])*1.0/N))!=T) mexErrMsgTxt("T must be integer\n");
Bcb = (uint8_t*)mxGetPr(prhs[1]);

//outputs
plhs[0] = mxCreateNumericMatrix(T,N,mxINT32_CLASS,mxREAL);
int* Shat = (int*)mxGetPr(plhs[0]);

#pragma omp parallel for
for(i=0;i<N;i++){
    if((i+1)%2000==0) mexPrintf("%d...",i+1);
    if((i+1)%20000==0) mexPrintf("\n");
    deal_one_sample(Bcb, Bcb+i*K, triplets+i*T, triplets+(i+N)*T, N, T, K, Shat+i*T);
}
mexPrintf("\n");

}