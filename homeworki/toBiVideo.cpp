#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	//input video and output video
	const string source = argv[1];
							//"D:\\FFOutput\\SampleInput.avi";
	double thresh = atof(argv[2]);
						 //140.0;
	const string output = argv[3];
							//"D:\\FFOutput\\SampleOutput.avi";

	//Load the input video
	VideoCapture inputVideo(source);
	if (!inputVideo.isOpened()){
		cout << "Could not open the input Video!" << endl;
		return -1;
	}

	//Get the Codec Type - Int form. Output video use the same codec type as input video.
	int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));

	Size S = Size((int)inputVideo.get(CAP_PROP_FRAME_WIDTH),
				(int)inputVideo.get(CAP_PROP_FRAME_HEIGHT));

	VideoWriter outputVideo;
	
	//Use twice as fast as the play speed
	double outputFPS = inputVideo.get(CAP_PROP_FPS) * 2;
	outputVideo.open(output, ex, outputFPS, S, false);
	
	if (!outputVideo.isOpened()){
		cout << "Could not open the output Video!" << endl;
		return -1;
	}

	Mat src, res;
	while (true){
		inputVideo >> src;
		if (src.empty())
			break;

		//Translate the frame to a binary image
		cvtColor(src, res, COLOR_BGR2GRAY);
		threshold(res, res, thresh, 255, THRESH_BINARY);

		//Add the text
		putText(res, "Kaixie Lv, 3120101867", Point(0, S.height-10), FONT_HERSHEY_DUPLEX, 1.0f, Scalar(255, 255, 255));

		imshow("Output Video", res);
		waitKey(1);
		outputVideo << res;
	}

	cout << "Finished." << endl;

	return 0;
}