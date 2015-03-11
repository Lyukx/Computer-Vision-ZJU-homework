#include <cv.h>
#include <highgui.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

double scale = 1;
CvHaarClassifierCascade *cascade = 0;
bool findFlag = true;
vector<string> names;
vector<string> labels_name;
CvRect* r = 0;//检测到的人脸图像

static Mat toGrayscale(InputArray _src) {
    Mat src = _src.getMat();
    // only allow one channel
    if(src.channels() != 1) {
        CV_Error(CV_StsBadArg, "Only Matrices with one channel are supported");
    }
    // create and return normalized image
    Mat dst;
    cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "指定了不存在的csv文件";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(toGrayscale(imread(path, 0)));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

static void read_names(){
	ifstream file("names.txt", ifstream::in);
	if (!file){
		string error_message = "不存在名字表";
		CV_Error(CV_StsBadArg, error_message);
	}
	string name, label, line;
	while (getline(file, line)){
		stringstream liness(line);
		getline(liness, name, ';');
		getline(liness, label);
		if (!name.empty() && !label.empty()){
			labels_name.push_back(label);
			names.push_back(name);
		}
	}
}

void cvText(IplImage* img, const char* text, int x, int y)
{
	CvFont font;
	double hscale = 1.0;
	double vscale = 1.0;
	int linewidth = 2;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, hscale, vscale, 0, linewidth);
	CvScalar textColor = cvScalar(0, 255, 255);
	CvPoint textPos = cvPoint(x, y);
	cvPutText(img, text, textPos, &font, textColor);
}

Mat findFace(IplImage *img){
	IplImage *gray, *small_img;
	Mat test;
	gray = cvCreateImage(cvSize(img->width, img->height), 8, 1);
	small_img = cvCreateImage(cvSize(cvRound(img->width / scale),
		cvRound(img->height / scale)), 8, 1);
	cvCvtColor(img, gray, CV_BGR2GRAY); // 彩色RGB图像转为灰度图像   
	cvResize(gray, small_img, CV_INTER_LINEAR);
	cvEqualizeHist(small_img, small_img); // 直方图均衡化   
	CvMemStorage *storage = cvCreateMemStorage(0);
	cvClearMemStorage(storage);
	if (cascade)
	{
		
		CvSeq* faces = cvHaarDetectObjects(small_img, cascade, storage,
			1.1, 2, 0 | CV_HAAR_DO_CANNY_PRUNING,
			cvSize(30, 30));
		
		if (faces->total == 0){
			findFlag = false;
			Mat M;
			return M;
		}

		r = (CvRect*)cvGetSeqElem(faces, 0); //我们只检测图像中的第一张脸   
		CvMat small_img_roi;
		CvPoint center;   
		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale); // 找出faces中心   
		center.y = cvRound((r->y + r->height*0.5)*scale);
		radius = cvRound((r->width + r->height)*0.25*scale);

		cvGetSubRect(small_img, &small_img_roi, *r);

		//截取检测到的人脸区域作为识别的图像  
		IplImage *result;
		CvRect roi;
		roi = *r;
		result = cvCreateImage(cvSize(r->width, r->height), img->depth, img->nChannels);
		cvSetImageROI(img, roi);
		// 创建子图像  
		cvCopy(img, result);
		cvResetImageROI(img);

		IplImage *resizeRes;
		CvSize dst_cvsize;
		//在我们的图像库中，标准的人脸图像大小为92*112
		dst_cvsize.width = (int)(92);
		dst_cvsize.height = (int)(112);
		resizeRes = cvCreateImage(dst_cvsize, result->depth, result->nChannels);
		
		cvResize(result, resizeRes, CV_INTER_NN);
		IplImage* img1 = cvCreateImage(cvGetSize(resizeRes), IPL_DEPTH_8U, 1);//创建目标图像    
		cvCvtColor(resizeRes, img1, CV_BGR2GRAY);

		CvScalar color = { 0, 0, 255 }; // 使用红色圈出人脸的大致位置
		cvCircle(img, center, radius, color, 3, 8, 0); // 从中心位置画圆，圈出脸部区域  

		test = img1;
	}
	cvReleaseImage(&gray);
	cvReleaseImage(&small_img);
	//img指针的释放在调用的地方做
	return test;
}

int main(int argc, const char *argv[]) {

    string fn_csv = "facerec_at_t.txt";

    vector<Mat> images;
    vector<int> labels;

	cout << "正在读入人脸训练集...";
    try {
        read_csv(fn_csv, images, labels);
		read_names();
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }
	cout << "done." << endl;

	cout << "正在加载分类器...";
	cascade = (CvHaarClassifierCascade*)cvLoad("haarcascade_frontalface_alt.xml", 0, 0, 0);
	if (!cascade){
		cout << "无法加载分类器！" << endl;
		return -1;
	}

    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
	cout << "done." << endl;

	cout << "请输入1，2，3选择使用Eigenface方式、Fisherface方式或LBPHface方式，输入其余值将同时使用三种方式并需要用户指定阈值" << endl;
	int method;
	cin >> method;
	Ptr<FaceRecognizer> model_eigen = createEigenFaceRecognizer();
	Ptr<FaceRecognizer> model_fisher = createFisherFaceRecognizer();
	Ptr<FaceRecognizer> model_lbph = createLBPHFaceRecognizer();

	if (method != 2 && method != 3){
		double t = (double)cvGetTickCount();
		cout << "正在进行训练（Eigenface方式）...";
		model_eigen->train(images, labels);
		cout << "done. ";		
		t = (double)cvGetTickCount() - t; //用于统计时间
		cout << "\t耗时（ms）：" << t / ((double)cvGetTickFrequency() * 1000) << endl;
	}

	if (method != 1 && method != 3){
		double t = (double)cvGetTickCount();
		cout << "正在进行训练（Fisherface方式）...";
		model_fisher->train(images, labels);
		cout << "done. ";
		t = (double)cvGetTickCount() - t; //用于统计时间
		cout << "\t耗时（ms）：" << t / ((double)cvGetTickFrequency() * 1000) << endl;
	}

	if (method != 2 && method != 1){
		double t = (double)cvGetTickCount();
		cout << "正在进行训练（LBPHface方式）...";
		model_lbph->train(images, labels);
		cout << "done. ";
		t = (double)cvGetTickCount() - t; //用于统计时间
		cout << "\t耗时（ms）：" << t / ((double)cvGetTickFrequency() * 1000) << endl;
	}

	cout << "请输入待测试图像的路径或输入quit退出系统：" << endl;
	string testSampleRoute;
	cin >> testSampleRoute;
	double threshold;
	if (method != 1 && method != 2 && method != 3){
		cout << "threshold:";
		cin >> threshold;
	}
	else{
		threshold = 100;
	}
	model_eigen->set("threshold", threshold);
	model_fisher->set("threshold", threshold);
	model_lbph->set("threshold", threshold);
	while (testSampleRoute != "quit"){
		IplImage *img = cvLoadImage(testSampleRoute.c_str(), CV_LOAD_IMAGE_UNCHANGED);
		Mat testSample = findFace(img);
		if (!findFlag){
			findFlag = true;
			cout << "无法在输入图像中检测到人脸！" << endl;
		}
		else{
			imwrite("test.jpg", testSample);
			int predictedLabel_e = -1;
			int predictedLabel_f = -1;
			int predictedLabel_l = -1;

			if (method != 2 && method != 3){
				double t = (double)cvGetTickCount();
				predictedLabel_e = model_eigen->predict(testSample);
				cout << "Eigenface Predicted class = " << predictedLabel_e;
				t = (double)cvGetTickCount() - t; //用于统计时间
				cout << "\t耗时（ms）：" << t / ((double)cvGetTickFrequency() * 1000) << endl;
			}
			if (method != 1 && method != 3){
				double t = (double)cvGetTickCount();
				predictedLabel_f = model_fisher->predict(testSample);
				cout << "Fisherface Predicted class = " << predictedLabel_f;
				t = (double)cvGetTickCount() - t; //用于统计时间
				cout << "\t耗时（ms）：" << t / ((double)cvGetTickFrequency() * 1000) << endl;
			}
			if (method != 2 && method != 1){
				double t = (double)cvGetTickCount();
				predictedLabel_l = model_lbph->predict(testSample);
				cout << "LBPHface Predicted class = " << predictedLabel_l;
				t = (double)cvGetTickCount() - t; //用于统计时间
				cout << "\t耗时（ms）：" << t / ((double)cvGetTickFrequency() * 1000) << endl;
			}

			//将人脸辨识结果写到原始图像中去
			string name[3] = { "", "", "" };
			//寻找是否存在于名字表内
			for (int i = 0; i < labels_name.size(); i++){
				if (atoi(labels_name.at(i).c_str()) == predictedLabel_e)
					name[0] = names[i];
				if (atoi(labels_name.at(i).c_str()) == predictedLabel_f)
					name[1] = names[i];
				if (atoi(labels_name.at(i).c_str()) == predictedLabel_l)
					name[2] = names[i];
			}
			if (name[0] == "" && predictedLabel_e != -1)
				name[0] = format(" %d ", predictedLabel_e);
			if (name[1] == "" && predictedLabel_f != -1)
				name[1] = format(" %d ", predictedLabel_f);
			if (name[2] == "" && predictedLabel_l != -1)
				name[3] = format(" %d ", predictedLabel_l);

			string text = name[0] + name[1] + name[2];
			cvText(img, text.c_str(), r->x + r->width*0.5, r->y);
			cvShowImage("Input image", img);
			imshow("Test image", testSample);
			waitKey(0);
			cvDestroyAllWindows();
			cvReleaseImage(&img);
		}
		cout << "请输入待测试图像的路径：" << endl;
		cin >> testSampleRoute;
		if (method != 1 && method != 2 && method != 3){
			cout << "threshold:";
			cin >> threshold;
			model_eigen->set("threshold", threshold);
			model_fisher->set("threshold", threshold);
			model_lbph->set("threshold", threshold);
		}
	}


    return 0;
}
