#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <NvInfer.h>
#include <cassert>

#include <cuda_profiler_api.h> 

#include "common/logger.h"
#include "dcn_v2.hpp" //! DCN plugin

// #include "gpu_sort.hpp"
#include "det_kernels.hpp"
#include "custom.hpp"
#include "ResizeBilinear.hpp"
#include "typeinfo"

using namespace std;
using namespace cv;
using namespace nvinfer1;

#define CHECK_CUDA(e) { if(e != cudaSuccess) { \
    printf("cuda failure: %s:%d: '%s'\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); \
        exit(0); \
    } \
}

struct NvInferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

int getBindingInputIndex(IExecutionContext* context) {
    return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
}

template <typename T>
void createBatchBuffers(T* buff[], T* data, const size_t len_per_batch, const int batch_num) {
    for(int i = 0; i < batch_num; ++i) {
        buff[i] = data + len_per_batch * i;
    }
}


int main(int argc, char* argv[]) {
    CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    /// read the serialized engine 
    string trt_file(argv[1]); 
    vector<char> trtModelStream_;
    size_t size(0);
    cout << "Loading engine file:" << trt_file << endl;
    ifstream file(trt_file, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream_.resize(size);
        file.read(trtModelStream_.data(), size);
        file.close();
    }
    cout << " size: " << size << endl;
    auto runtime = unique_ptr<IRuntime, NvInferDeleter>(createInferRuntime(gLogger));
    assert(runtime);
    auto engine = unique_ptr<ICudaEngine, NvInferDeleter>(runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr));
    if(!engine) {
        cerr << " Failed to create the engine from .trt file!" << endl;
    } else {
        cout << " Create the engine from " << trt_file << " successfully!" << endl;
    }
    /// an execution context holds additional memory to store intermediate activation values. an engine can have multiple contexts sharing the same weights for multi-tasks/streams
    auto context = unique_ptr<IExecutionContext, NvInferDeleter>(engine->createExecutionContext()); //Create some space to store intermediate activation values.
    if(!context) {
        cerr << " Failed to createExecutionContext!" << endl;
        exit(-1);
    }


    string video_name(argv[2]);
    VideoCapture capture(video_name);
    Mat frame;
    capture>> frame;
    if (!capture.isOpened()) {
        cout << "Can not open this video" << endl;
        return -1;
    }
    const int total_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
    cout << total_frames << endl;

    const int nframe = 5;
    //const int batch_num = total_frames / 5;
    const int batch_num = 1;
    const int up_scale = 4;
    const int net_h = 64;
    const int net_w = 64;
    const int net_oh = net_h * up_scale;
    const int net_ow = net_w * up_scale;


    const size_t input_size = net_h * net_w * 3 * 5 * batch_num; //! directly copy cv::Mat to GPU mem.
    const size_t output_size = net_oh * net_ow * 3 * batch_num;
    const size_t img_size = 540 * 960 * 3 * nframe * batch_num;
    // const size_t hm_size = net_oh * net_ow * num_classes * batch_num;
    // const size_t wh_size = net_oh * net_ow * 2 * batch_num;
    // const int det_len = 1 + K * 6;
    // const size_t det_size = det_len * batch_num;


    float* buffers[2];

    uint8_t* d_in;
   
    /// 
    //CHECK_CUDA(cudaMalloc((void**)&d_in, sizeof(uint8_t) * img_size));
    CHECK_CUDA(cudaMalloc((void**)&buffers[0], sizeof(float) * input_size));
    CHECK_CUDA(cudaMalloc((void**)&buffers[1], sizeof(float) * output_size));
    CHECK_CUDA(cudaMemset(buffers[0], 0 , sizeof(float)*input_size));



    // memory allocation for det images
    // string img_name(argv[2]);

    // Mat img = imread(img_name);
    // assert (img.data != NULL);

    // 1 * 3 * w * h -> 1* 15 * w * h



     cout <<"batch num:"<< batch_num <<endl;
     
     uint8_t* h_imgs[256];

     uint8_t ** d_imgs;
     //cudaMalloc((void**)&d_imgs, sizeof(uint8_t*) * 5 *batch_num); 
     createBatchBuffers<uint8_t>(h_imgs, d_in, 540*960*3, 5 * batch_num);
     cudaMemcpy(d_imgs, h_imgs, sizeof(uint8_t*)*5*batch_num, cudaMemcpyHostToDevice);


     const int K = total_frames / 5 / batch_num; 
     cout <<"K:"<< K << endl;
     //context->enqueue(batch_num, (void**)buffers, stream, nullptr);




     int cnt = 0;
     int flag1 = 0, flag2 = 0, flag3 = 0;
     for(int kk = 0;kk < K; kk++) {
	if (flag3 == 1) break;
     for (int i = 0; i < batch_num;i++) {
	if (flag2 == 1) break;
        for (int j = 0; j < 5; j++) {
            capture.read(frame); 
	    if (frame.empty()) {
                 cout << "quit!" << endl;
		// return -1;
		flag1 = 1;
		flag2 = 1;
		flag3 = 1;
            }
	    if (flag1 == 1) break;
	     

	    int idx = i *5 + j ;
	    //cout <<"idx:" << idx << "frame:" <<frame.rows <<", " << frame.cols <<endl;
	    cnt++;
	    size_t shift =  540 * 960 * 3 * idx; 
	    cout << "frame.data's type: " << typeid(frame.data).name() << endl;

            CHECK_CUDA(cudaMemcpyAsync(d_in+shift, frame.data, sizeof(uint8_t) * frame.rows * frame.cols * 3, cudaMemcpyHostToDevice, stream));
            //CHECK_CUDA(cudaMemcpyAsync(d_imgs[idx], frame.data, sizeof(uint8_t) * frame.rows * frame.cols * 3, cudaMemcpyHostToDevice, stream));
        }
    }
    //cuda_centernet_preprocess(batch_num, d_in, 15,  frame.rows, frame.cols,   
    //                                buffers[0], net_h, net_w, stream);

            //cout << " starting inference "<< "batch: " << kk << endl;
    // TensorRT execution is typically asynchronous, so enqueue the kernels on a CUDA stream
	
    // context->enqueue(batch_num, (void**)buffers, stream, nullptr);

    // cudaStreamSynchronize(stream);       
     }
     cout << "cnt = " << cnt << endl;



    for(int i = 0;i < 2; ++i) CHECK_CUDA(cudaFree(buffers[i]));
    CHECK_CUDA(cudaStreamDestroy(stream));


    return 0;
}
