#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <string>


using namespace cv;
using namespace std;

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
						hconcat(tp, tp, tp);
						vconcat(tp, tp, tp);
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

int main()
{
	//read MNIST iamge into OpenCV Mat vector
		vector<cv::Mat> vec;
		read_Mnist("train-images-idx3-ubyte", vec);
//		cout << vec.size() << endl;
		cout << vec[3] << endl;

//		for (int i = 0; i < vec.size(); i++) {
//			imshow("Window", vec[i]);
//		    waitKey(100);
//		}
//imshow("Window", vec[3]);
//waitKey(0);
	vector<unsigned char> arr;
	read_Mnist_Label("train-labels-idx1-ubyte", arr);
	/*for (int i = 0; i < arr.size(); ++i) {
		cout << arr[i] << " " << endl;
		if (i%50 == 0) cout << endl;
	}*/
	//cout << arr[0] << endl << arr[1] << endl;
	return 0;
}