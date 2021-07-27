#include <stdio.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"


#include "tensorflow/lite/kmcontext.h"

using namespace cv;
using namespace std;

/// Besides your regular TensorFlow Lite and flatbuffers library,
/// you must also compiled TensorFlow Lite from scratch by bazel
/// with the option GPU delegate set, before you can use the GPU delegates
/// see https://qengineering.eu/install-tensorflow-2-lite-on-jetson-nano.html
/// note also, it will not bring any speed improvement.
//#define GPU_DELEGATE      //remove comment to deploy GPU delegates

#ifdef GPU_DELEGATE
    #include "tensorflow/lite/delegates/gpu/delegate.h"
#endif // GPU_DELEGATE

#define TFLITE_MINIMAL_CHECK(x)									\
	if (!(x)) {													\
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);\
		exit(1);												\
	}

void DebugTensorflow();

int ReverseInt(int i)
{
		unsigned char ch1, ch2, ch3, ch4;
		ch1 = i & 255;
		ch2 = (i >> 8) & 255;
		ch3 = (i >> 16) & 255;
		ch4 = (i >> 24) & 255;
		return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(string filename, vector<cv::Mat>& vec) {
		ifstream file(filename, ios::binary);
		if (file.is_open())
		{
				int magic_number = 0;
				int number_of_images = 0;
				int n_rows = 0;
				int n_cols = 0;
				file.read((char*)& magic_number, sizeof(magic_number));
				magic_number = ReverseInt(magic_number);
				file.read((char*)& number_of_images, sizeof(number_of_images));
				number_of_images = ReverseInt(number_of_images);
				file.read((char*)& n_rows, sizeof(n_rows));
				n_rows = ReverseInt(n_rows);
				file.read((char*)& n_cols, sizeof(n_cols));
				n_cols = ReverseInt(n_cols);
				for (int i = 0; i < number_of_images; ++i)
				{
						cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
						for (int r = 0; r < n_rows; ++r)
						{
								for (int c = 0; c < n_cols; ++c)
								{
										unsigned char temp = 0;
										file.read((char*)& temp, sizeof(temp));
										tp.at<uchar>(r, c) = (int)temp;
								}
						}
						//hconcat(tp, tp, tp);
						//vconcat(tp, tp, tp);
						vec.push_back(tp);
				}
		}
        else {
            cout << "file open failed" << endl;
        }
}

void read_Mnist_Label(string filename, vector<unsigned char> &arr) {
	ifstream file(filename, ios::binary);
	if (file.is_open()) 
	{
		for (int i = 0; i < 10000; ++i) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			if (i > 7) {
				//cout << (int)temp << " ";
				arr.push_back((unsigned char)temp);
			}
		}
	}
	else {
        cout << "file open failed" << endl;
    }
}

void set_input(vector<cv::Mat> &mat, float* out, int num){
	/*cout << vec[0];
	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			cout << (int)(vec[0].at<unsigned char>(i,j)) << "\t";
		}
		cout << endl;
	}*/
	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			//cout << (int)(mat[0].at<unsigned char>(i,j)) / 255.0 << "\t";
			out[i * 28 + j] = (int)(mat[num].at<unsigned char>(i,j)) / 255.0;
			//cout << out[i*56 + j]*255.0 << "\t";
		}//cout << endl;
	}
	//cout << mat[num] << endl;
	/*for(int i = 0; i < 28*28*1; ++i) {
		out[i] = 0.1;
	}*/

}


int main(int argc,char ** argv)
{	
	if (argc != 2) {
		fprintf(stderr, "exe <tflite model>\n");
		return 1;
	}
	const char* filename = argv[1];

	// Load model
	std::cout << "==== START Load model ====\n\n";
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
	TFLITE_MINIMAL_CHECK(model != nullptr);
	std::cout << "\n==== END Load model ====\n\n\n\n";

    // Build the interpreter
	std::cout << "==== START Build the interpreter ====\n\n";
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
	std::unique_ptr<tflite::Interpreter> interpreter;
	builder(&interpreter);
	TFLITE_MINIMAL_CHECK(interpreter != nullptr);
	std::cout << "\n==== END Build the interpreter ====\n\n\n\n";
interpreter->AllocateTensors();

#ifdef GPU_DELEGATE
    std::cout << "==== START GPU_DELEGATE ====\n\n";
    TfLiteDelegate *MyDelegate = NULL;

    const TfLiteGpuDelegateOptionsV2 options = {
        .is_precision_loss_allowed = 1, //FP16,
        .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
        .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
        .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
        .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
		.experimental_flags = 4,
		.max_delegated_partitions = 5,
    };
    MyDelegate = TfLiteGpuDelegateV2Create(&options);

    if(interpreter->ModifyGraphWithDelegate(MyDelegate) != kTfLiteOk) {
        std::cerr << "ERROR: Unable to use delegate" << std::endl;
        return 0;
    }
	std::cout << "\n==== END GPU_DELEGATE ====\n\n\n\n";
#endif // GPU_DELEGATE

	std::cout << "==== START AllocateTensors() ====\n\n";
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
	std::cout << "\n==== END AllocateTensors() ====\n\n\n\n";

    //interpreter->SetAllowFp16PrecisionForFp32(true);
	
	//TFLITE_MINIMAL_CHECK(interpreter->SetNumThreads(12) == kTfLiteOk);


	vector<unsigned char> labels;
 	read_Mnist_Label("./mnist/t10k-labels-idx1-ubyte", labels);
	vector<cv::Mat> mat;
	read_Mnist("./mnist/t10k-images-idx3-ubyte", mat);

	vector<pair<int, float>> layer_and_ratio;
	//layer_and_ratio.push_back(make_pair(0, 0.75));
	//layer_and_ratio.push_back(make_pair(1, 0.75));
	//kmcontext.channelPartitioning(layer_and_ratio);
	
	kmcontext.printNodeDims();

	kmcontext.channelPartitioning("CONV_2D", 0.75);
	
	//DebugTensorflow();

	int accuracy = 0;
	int test_num = 1;
		clock_t start = clock();

	for (int num = 0; num < test_num; ++num) {
		set_input(mat, interpreter->typed_input_tensor<float>(interpreter->inputs()[0]), num);

		std::cout << "\n====START Invoke====\n\n";
			TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
		std::cout << "\n====END Invoke====\n\n";

		vector<float> results;
		for (int i = 0; i < 10; ++i) {
			results.push_back(interpreter->typed_output_tensor<float>(0)[i]);
			//cout << results[i] << std::endl;
		}

		if ((int)labels[num] == max_element(results.begin(), results.end()) - results.begin()){
			++accuracy;
		}
	}
	cout << "accuracy : " << accuracy / (float)test_num << endl;
	cout << (float)(clock()-start)/CLOCKS_PER_SEC << endl;


	DebugTensorflow();
		//std::cout << "\n====set_input====\n\n";

		//std::cout << "\n====get_output====\n\n";

		//cout << max_element(label.begin(), label.end()) - label.begin() << endl;
		//cout << (int)arr[num] << " " << max_element(label.begin(), label.end()) - label.begin() << endl;
		//cout << num << endl;

#ifdef GPU_DELEGATE
    interpreter.reset();
    TfLiteGpuDelegateV2Delete(MyDelegate);
#endif // GPU_DELEGATE

    return 0;
}


void DebugTensorflow() {
	std::cout << "DEBUG MODE" << endl;

	kmcontext.printNodeIndex();

	/*for (int execution_plan_index = 0;
		 execution_plan_index < kmcontext.execution_plan_->size(); execution_plan_index++) {
		int node_index = kmcontext.execution_plan_[0][execution_plan_index];
		TfLiteNode& node = kmcontext.nodes_and_registration_[0][node_index].first;
		const TfLiteRegistration& registration = kmcontext.nodes_and_registration_[0][node_index].second;
		cout << endl << GetOpName(registration) << endl;

		cout << "input_index  : ";
		for (int i = 0; i < node.inputs->size; ++i) {
			cout << node.inputs->data[i] << " ";
		} cout << endl;
		
		cout << "output_index : ";
		for (int i = 0; i < node.outputs->size; ++i) {
			cout << node.outputs->data[i] << " ";
		} cout << endl;
	}*/

	cout << "END : index" << endl;

//    *((int*)kmcontext.context_->tensors[7].dims+1) -= 1;
//	*((int*)kmcontext.context_->tensors[3].dims+1) -= 1;
//	*((int*)kmcontext.context_->tensors[8].dims+4) -= 1;

	kmcontext.printNodeDims();

	/*string input_shape_[3] = { "input",
								"filter",
								"bias" };
	
	string output_shape_[1] = { "output" };

	for (int execution_plan_index = 0;
		 execution_plan_index < kmcontext.execution_plan_->size(); execution_plan_index++) {
		int node_index = kmcontext.execution_plan_[0][execution_plan_index];
		TfLiteNode& node = kmcontext.nodes_and_registration_[0][node_index].first;
		const TfLiteRegistration& registration = kmcontext.nodes_and_registration_[0][node_index].second;
		cout << endl << GetOpName(registration) << endl;

		for (int i = 0; i < node.inputs->size; ++i) {
			int tensor_index = node.inputs->data[i];
			int* dims = (int*)kmcontext.context_->tensors[tensor_index].dims;
			cout << input_shape_[i] << "_shape :\t";
			for (int j = 1; j <= *dims; ++j) {
				cout << *(dims + j) << " ";
			} cout << endl;
		}

		for (int i = 0; i < node.outputs->size; ++i) {
			int tensor_index = node.inputs->data[i];
			int* dims = (int*)kmcontext.context_->tensors[tensor_index].dims;
			cout << output_shape_[i] << "_shape :\t";
			for (int j = 1; j <= *dims; ++j) {
				cout << *(dims + j) << " ";
			} cout << endl;
		}
	}*/
	cout << "END : dims" << endl;
	cout << "END" << endl;	
}
