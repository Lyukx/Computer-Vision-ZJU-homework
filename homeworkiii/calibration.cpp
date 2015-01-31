#include <cv.h>
#include <highgui.h>
#include <iostream>

using namespace std;

//Information of the chess board.
int board_w = 12;
int board_h = 12;
int board_n = board_w * board_h;
CvSize board_sz = cvSize(board_w, board_h);

//相机参数和畸变矩阵
CvMat *intrinsic_matrix = cvCreateMat(3, 3, CV_32FC1);
CvMat *distortion_coeffs = cvCreateMat(4, 1, CV_32FC1);
bool flag = false;//记录图片是否已经经过相机标定，如果没有则不能进行鸟瞰图转换

void calibration()
{
	IplImage *img = NULL;
	
	img = cvLoadImage("view.jpg");
	
	int count;
	CvPoint2D32f *corners = new CvPoint2D32f[board_n];
	cout << "正在寻找角点...";
	int found = cvFindChessboardCorners(img, board_sz, corners, &count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
	cout << "done." << endl;
	if (found == 0){
		cout << "Cannot find all the corners in the image!" << endl;
	}
	else{
		cout << "The corners have been all found." << endl; 
		cvNamedWindow("Calibration", 1);

		cvShowImage("Calibration", img);
		cvWaitKey(0);

		cout << "正在获取源图像灰度图...";
		IplImage *gray_img = cvCreateImage(cvGetSize(img), 8, 1);
		cvCvtColor(img, gray_img, CV_BGR2GRAY);
		cout << "done." << endl;
		cvShowImage("Calibration", gray_img);
		cvWaitKey(0);

		cout << "灰度图亚像素画...";
		cvFindCornerSubPix(gray_img, corners, count, cvSize(11, 11), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		cout << "done." << endl;

		cout << "输出结果图...";
		cvDrawChessboardCorners(img, board_sz, corners, count, found);
		cvShowImage("Calibration", img);
		cout << "done." << endl;

		//Start to calculate the parameter
		CvMat *image_points = cvCreateMat(board_n, 2, CV_32FC1);
		CvMat *object_points = cvCreateMat(board_n, 3, CV_32FC1);
		CvMat *point_counts = cvCreateMat(1, 1, CV_32SC1);
		intrinsic_matrix = cvCreateMat(3, 3, CV_32FC1);
		distortion_coeffs = cvCreateMat(4, 1, CV_32FC1);

		if (count == board_n){//Only use one picture to calculate.
			for (int i = 0; i < board_n; ++i){
				CV_MAT_ELEM(*image_points, float, i, 0) = corners[i].x;
				CV_MAT_ELEM(*image_points, float, i, 1) = corners[i].y;
				CV_MAT_ELEM(*object_points, float, i, 0) = i / board_w;
				CV_MAT_ELEM(*object_points, float, i, 1) = i % board_w;
				CV_MAT_ELEM(*object_points, float, i, 2) = 0.0f;
			}
		}
		CV_MAT_ELEM(*point_counts, int, 0, 0) = board_n;
		cout << "正在计算相机参数...";
		cvCalibrateCamera2(object_points, image_points, point_counts, cvGetSize(img), intrinsic_matrix, distortion_coeffs, NULL, NULL, 0);
		cout << "done." << endl;
		cout << "相机内参矩阵：" << endl;
		cout << CV_MAT_ELEM(*intrinsic_matrix, float, 0, 0) << "\t" << CV_MAT_ELEM(*intrinsic_matrix, float, 0, 1) << "\t" << CV_MAT_ELEM(*intrinsic_matrix, float, 0, 2) << endl;
		cout << CV_MAT_ELEM(*intrinsic_matrix, float, 1, 0) << "\t" << CV_MAT_ELEM(*intrinsic_matrix, float, 1, 1) << "\t" << CV_MAT_ELEM(*intrinsic_matrix, float, 1, 2) << endl;
		cout << CV_MAT_ELEM(*intrinsic_matrix, float, 2, 0) << "\t" << CV_MAT_ELEM(*intrinsic_matrix, float, 2, 1) << "\t" << CV_MAT_ELEM(*intrinsic_matrix, float, 2, 2) << endl;
		cout << "相机内参：" << endl;
		cout << "x轴焦距:" << CV_MAT_ELEM(*intrinsic_matrix, float, 0, 0) << endl;
		cout << "y轴焦距:" << CV_MAT_ELEM(*intrinsic_matrix, float, 1, 1) << endl;
		cout << "图像平面坐标系参考点： (" << CV_MAT_ELEM(*intrinsic_matrix, float, 0, 2) << ", " << CV_MAT_ELEM(*intrinsic_matrix, float, 1, 2) << ")" << endl;
		
		cout << "相机畸变系数矩阵：" << endl;
		cout << "[k1, k2, p1, p2, k3] k为径向畸变系数，p为切向畸变系数" << endl;
		cout << CV_MAT_ELEM(*distortion_coeffs, float, 0, 0) << "\t" << CV_MAT_ELEM(*distortion_coeffs, float, 1, 0) << "\t" << CV_MAT_ELEM(*distortion_coeffs, float, 2, 0) << "\t" << CV_MAT_ELEM(*distortion_coeffs, float, 3, 0) << endl;


		IplImage* mapx = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
		IplImage* mapy = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
		cvInitUndistortMap(
			intrinsic_matrix,
			distortion_coeffs,
			mapx,
			mapy
			);
		cvNamedWindow("Undistort");
		IplImage *t = cvCloneImage(img);
		cvShowImage("Raw Video", img); // Show raw image
		cvRemap(t, img, mapx, mapy);     // Undistort image
		cvReleaseImage(&t);
		cvShowImage("Undistort", img);     // Show corrected image

		cvWaitKey(0);
		cvDestroyAllWindows();
		cvReleaseImage(&img);
		cvReleaseImage(&gray_img);
		cvReleaseImage(&t);
		cvReleaseImage(&mapx);
		cvReleaseImage(&mapy);
		flag = true;
	}
}

void birdview(){
	IplImage *image = NULL;
	image = cvLoadImage("view.jpg");
	IplImage *gray_img = NULL;
	gray_img = cvCreateImage(cvGetSize(image), 8, 1);
	cvCvtColor(image, gray_img, CV_BGR2GRAY);

	IplImage* mapx = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	IplImage* mapy = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	cvInitUndistortMap(
		intrinsic_matrix,
		distortion_coeffs,
		mapx,
		mapy
		);
	IplImage *t = cvCloneImage(image);
	cvRemap(t, image, mapx, mapy);

	CvPoint2D32f* corners = new CvPoint2D32f[board_n];
	int corner_count = 0;
	int found = cvFindChessboardCorners(
		image,
		board_sz,
		corners,
		&corner_count,
		CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS
		);
	if (!found){
		printf("Couldn't aquire checkerboard on sample, only found %d of %d corners\n",
			corner_count, board_n);
	}
	else{
		cvFindCornerSubPix(gray_img, corners, corner_count,
			cvSize(11, 11), cvSize(-1, -1),
			cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

		CvPoint2D32f objPts[4], imgPts[4];
		objPts[0].x = 0;			objPts[0].y = 0;
		objPts[1].x = board_w - 1;	objPts[1].y = 0;
		objPts[2].x = 0;			objPts[2].y = board_h - 1;
		objPts[3].x = board_w - 1;	objPts[3].y = board_h - 1;
		imgPts[0] = corners[0];
		imgPts[1] = corners[board_w - 1];
		imgPts[2] = corners[(board_h - 1)*board_w];
		imgPts[3] = corners[(board_h - 1)*board_w + board_w - 1];

		cvCircle(image, cvPointFrom32f(imgPts[0]), 9, CV_RGB(0, 0, 255), 3);
		cvCircle(image, cvPointFrom32f(imgPts[1]), 9, CV_RGB(0, 255, 0), 3);
		cvCircle(image, cvPointFrom32f(imgPts[2]), 9, CV_RGB(255, 0, 0), 3);
		cvCircle(image, cvPointFrom32f(imgPts[3]), 9, CV_RGB(255, 255, 0), 3);

		CvMat *H = cvCreateMat(3, 3, CV_32F);
		CvMat *H_invt = cvCreateMat(3, 3, CV_32F);
		cvGetPerspectiveTransform(objPts, imgPts, H);

		float Z = 20;
		int key = 0;
		IplImage *birds_image = cvCloneImage(image);
		cvNamedWindow("Birds_Eye");

		cout << "Press 'u' to look up and 'd' to look down. Press esc to end." << endl;
		while (key != 27) {//escape key stops
			CV_MAT_ELEM(*H, float, 2, 2) = Z;
			cvWarpPerspective(image, birds_image, H,
				CV_INTER_LINEAR + CV_WARP_INVERSE_MAP + CV_WARP_FILL_OUTLIERS);
			cvShowImage("Birds_Eye", birds_image);
			key = cvWaitKey();
			if (key == 'u') Z += 0.5;
			if (key == 'd') Z -= 0.5;
		}
		cvSaveImage("birdseye-view.jpg", birds_image);
		cvDestroyAllWindows();
		cvReleaseImage(&image);
		cvReleaseImage(&gray_img);
		cvReleaseImage(&t);
		cvReleaseImage(&mapx);
		cvReleaseImage(&mapy);
		cvReleaseImage(&birds_image);
	}
}

void help()
{
	cout << "Key in command to start this demo system." << endl;
	cout << "cali: Applying calibration of the view.jpg image." << endl;
	cout << "bv: Translate the view.jpg to a bird-view. Should apply calibration first." << endl;
}

int main()
{
	string cmd;
	help();
	while (true){
		cout << "user> ";
		cin >> cmd;
		if (cmd == "cali"){
			calibration();
		}
		else if (cmd == "bv"){
			if (!flag)
				cout << "Please apply calibration first" << endl;
			else
				birdview();
		}
		else if (cmd == "quit"){
			break;
		}
		else{
			cout << "Illegal command!" << endl;
		}
	}
	return 0;
}