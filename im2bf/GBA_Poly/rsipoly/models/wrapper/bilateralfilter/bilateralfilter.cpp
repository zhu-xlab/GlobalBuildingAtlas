#include "bilateralfilter.hpp"


void initializePermutohedral(float * image, int H, int W, float sigmargb, float sigmaxy, Permutohedral & lattice_){
    float * features = new float[H * W * 5];
	for( int j=0; j<H; j++ ){
		for( int i=0; i<W; i++ ){
		    int idx = j*W + i;
			features[idx*5+0] = float(i) / sigmaxy;
			features[idx*5+1] = float(j) / sigmaxy;
			features[idx*5+2] = float(image[0*W*H + idx]) / sigmargb;
			features[idx*5+3] = float(image[1*W*H + idx]) / sigmargb;
			features[idx*5+4] = float(image[2*W*H + idx]) / sigmargb;
		}
    }
    
    lattice_.init( features, 5, H * W );
    delete [] features;
}


void bilateralfilter(float * image, int len_image, float * in, int len_in, float * out, int len_out,
                              int H, int W, float sigmargb, float sigmaxy){
	Permutohedral lattice;
	initializePermutohedral(image, H, W, sigmargb, sigmaxy, lattice);
    // number of classes
    int K = len_in/W/H;
    
    float * out_p = new float[W*H];
    float * in_p = new float[W*H];
    for(int k=0;k<K;k++){
        for(int i=0;i<W*H;i++)
            in_p[i] = in[i+k*W*H];
        lattice.compute(out_p, in_p, 1);
        for(int i=0;i<W*H;i++)
            out[i+k*W*H] = out_p[i];
    }
    delete [] out_p;
    delete [] in_p;
}

void bilateralfilter_batch(float * images, int len_images, float * ins, int len_ins, float * outs, int len_outs,
                              int N, int K, int H, int W, float sigmargb, float sigmaxy){
    
    const int maxNumThreads = omp_get_max_threads();
    //printf("Maximum number of threads for this machine: %i\n", maxNumThreads);
    omp_set_num_threads(std::min(maxNumThreads,N));
    
    #pragma omp parallel for
    for(int n=0;n<N;n++){
        bilateralfilter(images+n*3*H*W, 3*H*W, ins+n*K*H*W, K*H*W, outs+n*K*H*W, K*H*W,
                              H, W, sigmargb, sigmaxy);
    }
    //printf("parallel for\n");
}

