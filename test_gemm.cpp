#include <iostream>
#include <sys/time.h>
#include <cmath>
#include <stdlib.h>
#include <future>
#include <mutex>

typedef float DType;
std::mutex m;
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

//gemm with tiling (intput1 tile row major, input2 tile col major)
//In this case, we assume that there is no need for padding
//Performance is not good, may be there is too much for loop.
void gemm_O4(DType* input1, DType* input2, DType* output, int M, int N, int K, int tiling_m, int tiling_n, int tiling_k)
{
    int TM = M/tiling_m;
    int TN = N/tiling_n;
    int TK = K/tiling_k;
    for(int m=0;m<M/tiling_m;m++)
    {
        for(int n=0;n<N/tiling_n;n++)
        {
            for(int k=0;k<K/tiling_k;k++)
            {
                for(int tm=0;tm<tiling_m;tm++)
                {
                    for(int tn=0;tn<tiling_n;tn++)
                    {
                        for(int tk=0;tk<tiling_k;tk++)
                        {
                            int startpos_1 = m*TK*tiling_m*tiling_k+k*tiling_m*tiling_k;
                            int startpos_2 = n*TK*tiling_k*tiling_n+k*tiling_k*tiling_n;
                            int startpos_o = m*TN*tiling_m*tiling_n+n*tiling_m*tiling_n;
                            //std::cout<<"index: "<<startpos_2+tn*tiling_k+tk<<std::endl;
                            //std::cout<<"index: "<<m<<","<<n<<","<<k<<","<<startpos_2<<","<<tn*tiling_k+tk<<std::endl;
                            output[startpos_o+tm*tiling_n+tn]+=input1[startpos_1+tm*tiling_k+tk]*input2[startpos_2+tn*tiling_k+tk];
                        }
                    }
                }
            }
        }
    }
}



//gemm with naive thread
void getKsum(DType* input1, DType* input2, DType *output, int K)
{
    DType sum,sum1,sum2,sum3,sum4;
    sum = sum1 = sum2 = sum3 = sum4 = 0;
    /*
    //warning!!!: here may cause error, need to modify!
    for(int i=0;i<K;i+=4)
    {
        sum1+=input1[i]*input2[i];
        sum2+=input1[i+1]*input2[i+1];
        sum3+=input1[i+2]*input2[i+2];
        sum4+=input1[i+3]*input2[i+3];
    }
    return sum1+sum2+sum3+sum4;
    */
    //warning!!!: here may cause error, need to modify!
    for(int i=0;i<K;i++)
    {
        *output+=input1[i]*input2[i];
    }
}
void gemm_O5_Async(DType* input1, DType* input2, DType *output, int M, int N, int K, int thM, int thN)
{
    //input1 row-major, input2 col-major
    int threadM = thM;
    int threadN = thN;
    for(int i=0;i<M*N;i+=threadM*threadN)
    {
        std::future<void> f[threadM*threadN];
        for(int j = 0;j<threadM*threadN;j++)
        {
            f[j] = std::async(std::launch::async, &getKsum, input1+((i+j)/N)*K, input2+((i+j)%N)*K,output+i+j,K);
        }
    }
    
}

void gemm_O5_Thread(DType* input1, DType* input2, DType *output, int M, int N, int K, int thM, int thN)
{
    //input1 row-major, input2 col-major
    int threadM = thM;
    int threadN = thN;
    for(int i=0;i<M*N;i+=threadM*threadN)
    {
        std::thread t[threadM*threadN];
        for(int j = 0;j<threadM*threadN;j++)
        {
            t[j] = std::thread(getKsum, input1+((i+j)/N)*K, input2+((i+j)%N)*K,output+i+j,K);
        }
        for(int j = 0;j<threadM*threadN;j++)
        {
            t[j].join();
        }
    } 
}

void getKsum_Kernel(DType* input1, DType* input2, DType *output, int K, int threads)
{
    DType sum[threads];
    std::thread thr[threads];
    int k = K/threads;
    for(int i=0;i<threads;i++)
    {
        thr[i] = std::thread(getKsum, input1+i*k, input2+i*k, sum+i, k);
    }
    for(int i=0;i<threads;i++)
    {
        thr[i].join();
        *output+=sum[i];
    }
}
void gemm_O5_KThread(DType* input1, DType* input2, DType *output, int M, int N, int K, int threads)
{
    //input1 row-major, input2 col-major
    for(int i=0;i<M;i++)
    {
        for(int j = 0;j<N;j++)
        {
            getKsum_Kernel(input1+i*K, input2+j*K, output+i*N+j, K, threads);
        }
    } 
}


void gemm_O5_MThread(DType* input1, DType* input2, DType *output, int M, int N, int K, int threads)
{
    //input1 row-major, input2 col-major
    int thm = M/threads;
    std::thread thr[threads];
    for(int i=0;i<threads;i++)
    {
        thr[i] = std::thread(gemm_O2, input1+i*thm*K, input2, output+i*thm*N, thm, N, K);
    } 
    for(int i = 0;i<threads;i++)
    {
        thr[i].join();
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
    int TM = 4;
    int TN = 4;
    int TK = 4;
    if(argc==5)
        TM = atoi(argv[4]);
    if(argc==6)
    {   
        TM = atoi(argv[4]);
        TN = atoi(argv[5]);
    }
    if(argc==7)
    {
        TM = atoi(argv[4]);
        TN = atoi(argv[5]);
        TK = atoi(argv[6]);
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
    //gemm_O4(input1, input2, output, M, N, K, TM, TN, TK);
    gemm_O5_MThread(input1, input2, output, M, N, K, TM);
    //gemm_O2(input1, input2, output, M,N,K);
    gettimeofday(&end, NULL);
    float elapsed_time = (float)(end.tv_sec-start.tv_sec)+(float)(end.tv_usec-start.tv_usec)/std::pow(10,6);
    std::cout<<"Elapsed time: "<<elapsed_time<<", output: "<<output[0]<<std::endl;

    delete input1;
    delete input2;
    delete output;
}
