#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

class cellAttr{
public:
	cellAttr(int areaIn, int perimeterIn, CvPoint centerIn, float angleIn){
		area = areaIn;
		perimeter = perimeterIn;
		center = centerIn;
		angle = angleIn;
	}
	int area;
	int perimeter;
	CvPoint center;
	float angle;
};

//apply Otsu method to get the thread
int otsu(const IplImage *src_image)
{
	double sum = 0.0;
	double w0 = 0.0;
	double w1 = 0.0;
	double u0_temp = 0.0;
	double u1_temp = 0.0;
	double u0 = 0.0;
	double u1 = 0.0;
	double delta_temp = 0.0;
	double delta_max = 0.0;

	int pixel_count[256] = { 0 };
	float pixel_pro[256] = { 0 };
	int threshold = 0;
	uchar* data = (uchar*)src_image->imageData;
	for (int i = 0; i < src_image->height; i++){
		for (int j = 0; j < src_image->width; j++){
			pixel_count[(int)data[i * src_image->width + j]]++;
			sum += (int)data[i * src_image->width + j];
		}
	} 
	for (int i = 0; i < 256; i++){
		pixel_pro[i] = (float)pixel_count[i] / (src_image->height * src_image->width);
	}
	for (int i = 0; i < 256; i++){
		w0 = w1 = u0_temp = u1_temp = u0 = u1 = delta_temp = 0;
		for (int j = 0; j < 256; j++){
			if (j <= i){
				w0 += pixel_pro[j];
				u0_temp += j * pixel_pro[j];
			}
			else{
				w1 += pixel_pro[j];
				u1_temp += j * pixel_pro[j];
			}
		}
		u0 = u0_temp / w0;
		u1 = u1_temp / w1;
		delta_temp = (float)(w0 *w1* pow((u0 - u1), 2));
		if (delta_temp > delta_max){
			delta_max = delta_temp;
			threshold = i;
		}
	}

	return threshold;
}

void analyzeCell(IplImage *cell)
{
	cvShowImage("Step1, load the grey image", cell);

	//Translate the grey image to bi-image.
	IplImage *tempImg;
	IplImage *biImg;
	IplConvKernel *element;//Used to apply open/erode/dilate opration
	tempImg = cvCreateImage(cvGetSize(cell), cell->depth, cell->nChannels);
	biImg = cvCreateImage(cvGetSize(cell), cell->depth, cell->nChannels);
	element = cvCreateStructuringElementEx(4, 4, 1, 1, CV_SHAPE_ELLIPSE, 0);
	cvDilate(cell, cell, element, 1);

	cvThreshold(cell, biImg, otsu(cell), 255, CV_THRESH_BINARY_INV);
	cvSmooth(biImg, biImg, CV_MEDIAN, 7, 0, 0, 0);
	cvShowImage("Step2, bi-image", biImg);

	//Count the number of cells
	CvMemStorage *stor = 0;
	CvSeq * cont = 0;
	CvSeq * a_contour = 0;
	int cellNum = 0;
	stor = cvCreateMemStorage(0);
	cont = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), stor);
	a_contour = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), stor);

	cellNum = cvFindContours(biImg, stor, &cont, sizeof(CvContour),
		CV_RETR_TREE, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));
	//Output the number of cells
	cout << "Total cell number: " << cellNum << endl;

	//Analyze the attributes of all elements
	IplImage *edge;
	edge = cvCreateImage(cvGetSize(cell), cell->depth, cell->nChannels);
	cvThreshold(edge, edge, 0, 255, CV_THRESH_BINARY);
	int areaSum = 0;
	vector<cellAttr> cells;
	for (; cont; cont = cont->h_next){
		cvDrawContours(edge, cont, CV_RGB(0, 255, 0), CV_RGB(255, 0, 0), 0, 1, 8, cvPoint(0, 0));
		int area = fabs(cvContourArea(cont, CV_WHOLE_SEQ));
		int perimeter = fabs(cvArcLength(cont, CV_WHOLE_SEQ));
		if (area > 50){
			areaSum += area;

			//Get the fitting minimal box(rectangle)
			CvBox2D box = cvMinAreaRect2(cont, stor);
			CvPoint center = CvPoint(box.center.x, box.center.y);
			float angle = box.angle;

			//Push the attributes of current cell into the cell vector
			cellAttr theCell(area, perimeter, center, angle);
			cells.push_back(theCell);
		}
	}
	//Calculate the average area
	int areaAve = areaSum / cellNum;

	int areaMin = areaAve;
	int areaMax = areaAve;
	int perimeterMax, perimeterMin;
	CvPoint minCell, maxCell;
	float maxAngle, minAngle;

	//Find the max/min cell
	for (int i = 0; i < cells.size(); i++){
		if (cells[i].area > areaMax){
			areaMax = cells[i].area;
			perimeterMax = cells[i].perimeter;
			maxCell = cells[i].center;
			maxAngle = cells[i].angle;
		}
		if (cells[i].area < areaMin){
			areaMin = cells[i].area;
			perimeterMin = cells[i].perimeter;
			minCell = cells[i].center;
			minAngle = cells[i].angle;
		}
	}

	cvCircle(edge, maxCell, 20, cvScalar(0, 0, 0), 4);
	cvCircle(edge, minCell, 5, cvScalar(0, 0, 0), 2);

	cout << "Average area: " << areaAve << endl;
	cout << "Biggest cell's area: " << areaMax << endl;
	cout << "Biggest cell's perimeter: " << perimeterMax << endl;
	cout << "Biggest cell's center: (" << maxCell.x << ", " << maxCell.y << ")" << endl;
	cout << "Biggest cell's angle: " << maxAngle << endl;
	cout << "Smallest cell's area: " << areaMin << endl;
	cout << "Smallest cell's perimeter: " << perimeterMin << endl;
	cout << "Smallest cell's center: (" << minCell.x << ", " << minCell.y << ")" << endl;
	cout << "Smallest cell's angle: " << minAngle << endl;

	cvShowImage("Step 3, find the max/min cell", edge);
}

int main(int argc, char *argv[])
{
	//Load the images.
	IplImage *cell1;
	IplImage *cell2;
	IplImage *cell3;

	//Load the images as grey-scale maps.
	cell1 = cvLoadImage("cell1.bmp", 0);
	cell2 = cvLoadImage("cell2.jpg", 0);
	cell3 = cvLoadImage("cell3.jpg", 0);
	
	//First cell image
	analyzeCell(cell1);
	cvWaitKey();

	//Second cell image
	analyzeCell(cell2);
	cvWaitKey();

	//Third cell image
	analyzeCell(cell3);
	cvWaitKey();
	return 0;
}