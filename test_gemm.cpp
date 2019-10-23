#include <iostream>
#include <sys/time.h>
#include <cmath>
#include <stdlib.h>

typedef float DType;

//naive gemm (all row-major)
void gemm(DType* input1, DType* input2, DType* output, int M, int N, int K)
{
    //input1 [M, K]  input2 [K, N] row-major
    for(int m=0;m<M;m++)
    {
        for(int n=0;n<N;n++)
        {
            DType result = 0;
            for(int k=0;k<K;k++)
            {
                result+=input1[m*K+k]*input2[k*N+n];
            }
            output[m*N+n] = result;
        }
    }
}

//gemm with naive cache policy
void gemm_O1(DType* input1, DType* input2, DType* output, int M, int N, int K)
{
    //input1 [M, K], input2 [K, N] raw-major
    for(int m=0;m<M;m++)
    {
        for(int k=0;k<K;k++)
        {
            for(int n=0;n<N;n++)
            {
                output[m*N+n]+=input1[m*K+k]*input2[k*N+n];
            }
        }
    }
}

//gemm with naive cache policy (row-major & col-major)
void gemm_O2(DType* input1, DType* input2, DType* output, int M, int N, int K)
{
    //input1 [M, K],row-major, input2 [K, N] col-major
    for(int m=0;m<M;m++)
    {
        for(int n=0;n<N;n++)
        {
            DType result = 0;
            for(int k=0;k<K;k++)
            {
                result+=input1[m*K+k]*input2[n*K+k];
            }
            output[m*N+n] = result;
        }
    }
}

//gemm with naive tiling (row-major & col-major)
void gemm_O3(DType* input1, DType* input2, DType* output, int M, int N, int K, int tiling_h, int tiling_w)
{
    //input1 [M, K] row-major, input2 [K, N] col-major
    for(int m=0;m<M;m+=tiling_h)
    {
        for(int n=0;n<N;n+=tiling_w)
        {
            for(int th = 0;th<tiling_h;th++)
            {
                for(int tw = 0; tw<tiling_w;tw++)
                {
                    DType result = 0;
                    for(int k = 0; k<K; k++)
                    {
                        result+=input1[(m+th)*N+k]*input2[(n+tw)*K+k];
                    }
                    output[m*N+n] = result;
                }
            }
        }
    }
}

void parseArgs(int argc, char** argv)
{
    //if(argc)
}
int main(int argc, char* argv[])
{
    if(argc<4)
    {
        std::cout<<"Usage: ./demo M N K (tiling_w tiling_h)"<<std::endl;
        return -1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int TH = 4;
    int TW = 4;
    if(argc>4)
    {   
        TH = atoi(argv[4]);
        TW = atoi(argv[5]);
    }
    
    DType* input1 = new DType[M*K];
    DType* input2 = new DType[K*N];
    DType* output = new DType[M*N];
    for(int i = 0;i<M*K;i++)
    {
        input1[i] = 0.5;
    }
    for(int i = 0;i<K*N;i++)
    {
        input2[i] = 0.5;
    }
    for(int i = 0;i<M*N;i++)
    {
        output[i] = 0;
    }
    timeval start, end;
    gettimeofday(&start, NULL);
    gemm_O3(input1, input2, output, M, N, K,16,16);
    gettimeofday(&end, NULL);
    float elapsed_time = (float)(end.tv_sec-start.tv_sec)+(float)(end.tv_usec-start.tv_usec)/std::pow(10,6);
    std::cout<<"Elapsed time: "<<elapsed_time<<", output: "<<output[0]<<std::endl;

    delete input1;
    delete input2;
    delete output;
}
