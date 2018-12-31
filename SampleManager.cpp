#include "SampleManager.hpp"
#include "CudaModule.cuh"
#include "Detector.hpp"

#define COLOR_DISTANCE	100

#define RAND(a) (rand() % a)
#define RAND_POINT (cv::Point2i(rand() % IMG_WIDTH, rand() % IMG_HEIGHT))
#define RAND_CENTER (cv::Point2i(rand() % (IMG_WIDTH - 100) + 50, rand() % (IMG_HEIGHT - 100) + 50))
#define RAND_COLOR (cv::Scalar(rand() % 100, rand() % 100, rand() % 100))
#define RAND_CIRCLE_COLOR (cv::Scalar(rand() % 100 + COLOR_DISTANCE, rand() % 100 + COLOR_DISTANCE, rand() % 100 + COLOR_DISTANCE))
#define RAND_RADIUS (rand() % (R_MAX - R_MIN + 1) + R_MIN)
#define EU_DIST(a, b) (sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)))

#define LINE_THICKNESS 2

#define COMPRESSION_QUALITY 10

SampleManager::SampleManager(char* file_path_root)
{
	this->file_path_root = file_path_root;
	img_data = (uchar*)malloc(IMG_HEIGHT * IMG_WIDTH * SAMPLE_SIZE);
	binarray_data = (int*)malloc(32768 * SAMPLE_SIZE * sizeof(int));
}


void SampleManager::generateSamples()
{
	std::vector<cv::Rect> rect_list;
	char file_path[512];
	double avg = 0.0;
	double stdev = 50.0;
	cv::Mat noise(IMG_HEIGHT, IMG_WIDTH, CV_8UC3);
	std::vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
	compression_params.push_back(COMPRESSION_QUALITY);

	srand(0);

	printf("\nStarted to generate samples...\n");
	for(int i = 0; i < SAMPLE_SIZE; i++)
	{
		printf("%d.", i + 1);
		cv::Mat img(IMG_HEIGHT, IMG_WIDTH, CV_8UC3, cv::Scalar(100, 100, 100));
		rect_list.clear();
		rect_list.push_back(cv::Rect(RAND_POINT, RAND_POINT));
		rect_list.push_back(cv::Rect(RAND_POINT, RAND_POINT));
		rect_list.push_back(cv::Rect(RAND_POINT, RAND_POINT));
		rect_list.push_back(cv::Rect(RAND_POINT, RAND_POINT));
		rect_list.push_back(cv::Rect(RAND_POINT, RAND_POINT));
		rect_list.push_back(cv::Rect(RAND_POINT, RAND_POINT));
		rect_list.push_back(cv::Rect(RAND_POINT, RAND_POINT));
		rect_list.push_back(cv::Rect(RAND_POINT, RAND_POINT));
		std::sort(rect_list.begin(), rect_list.end(), [](cv::Rect a, cv::Rect b) { return a.area() > b.area(); });
		cv::rectangle(img, rect_list[0], RAND_COLOR, -1);
		cv::rectangle(img, rect_list[1], RAND_COLOR, -1);
		cv::rectangle(img, rect_list[2], RAND_COLOR, -1);
		cv::rectangle(img, rect_list[3], RAND_COLOR, -1);
		cv::rectangle(img, rect_list[4], RAND_COLOR, -1);
		cv::rectangle(img, rect_list[5], RAND_COLOR, -1);
		cv::rectangle(img, rect_list[6], RAND_COLOR, -1);
		cv::rectangle(img, rect_list[7], RAND_COLOR, -1);
		cv::line(img, RAND_POINT, RAND_POINT, RAND_COLOR, LINE_THICKNESS);
		cv::line(img, RAND_POINT, RAND_POINT, RAND_COLOR, LINE_THICKNESS);
		cv::line(img, RAND_POINT, RAND_POINT, RAND_COLOR, LINE_THICKNESS);
		cv::line(img, RAND_POINT, RAND_POINT, RAND_COLOR, LINE_THICKNESS);
		cv::line(img, RAND_POINT, RAND_POINT, RAND_COLOR, LINE_THICKNESS);
		cv::line(img, RAND_POINT, RAND_POINT, RAND_COLOR, LINE_THICKNESS);
		cv::line(img, RAND_POINT, RAND_POINT, RAND_COLOR, LINE_THICKNESS);
		cv::line(img, RAND_POINT, RAND_POINT, RAND_COLOR, LINE_THICKNESS);
		cv::Point2i centers[4];
		cv::Point2i cur_center;
		centers[0] = RAND_CENTER;
		cur_center = RAND_CENTER;
		while(EU_DIST(centers[0], cur_center) < R_MAX + R_MAX + 4)
			cur_center = RAND_CENTER;
		centers[1] = cur_center;
		while(EU_DIST(centers[0], cur_center) < R_MAX + R_MAX + 4 ||
			  EU_DIST(centers[1], cur_center) < R_MAX + R_MAX + 4)
			cur_center = RAND_CENTER;
		centers[2] = cur_center;
		while(EU_DIST(centers[0], cur_center) < R_MAX + R_MAX + 4 ||
			  EU_DIST(centers[1], cur_center) < R_MAX + R_MAX + 4 ||
			  EU_DIST(centers[2], cur_center) < R_MAX + R_MAX + 4)
			cur_center = RAND_CENTER;
		centers[3] = cur_center;
		cv::circle(img, centers[0], RAND_RADIUS, RAND_CIRCLE_COLOR, -1);
		cv::circle(img, centers[1], RAND_RADIUS, RAND_CIRCLE_COLOR, -1);
		cv::circle(img, centers[2], RAND_RADIUS, RAND_CIRCLE_COLOR, -1);
		cv::circle(img, centers[3], RAND_RADIUS, RAND_CIRCLE_COLOR, -1);
		labels.push_back(centers[0]);
		labels.push_back(centers[1]);
		labels.push_back(centers[2]);
		labels.push_back(centers[3]);

		cv::randn(noise, cv::Scalar(avg), cv::Scalar(stdev));
		cv::addWeighted(img, 1.0, noise, 1.0, 0.0, img);

		sprintf_s(file_path, "%soriginal\\%d.jpg", file_path_root, i + 1);
		cv::imwrite(file_path, img, compression_params);
	}
	printf("done!\n");


	FILE* fp;
	sprintf_s(file_path, "%slabel.csv", file_path_root);
	fopen_s(&fp, file_path, "wt");
	for(int i = 0; i < SAMPLE_SIZE; i++)
	{
		fprintf_s(fp, "%d|%d,%d|%d,%d|%d,%d|%d%s",
				  labels[i * 4].x, labels[i * 4].y,
				  labels[i * 4 + 1].x, labels[i * 4 + 1].y,
				  labels[i * 4 + 2].x, labels[i * 4 + 2].y,
				  labels[i * 4 + 3].x, labels[i * 4 + 3].y, (i == SAMPLE_SIZE - 1) ? "" : "\n");
	}
	fclose(fp);
}


void SampleManager::loadSamples()
{
	char file_path[512];
	cv::Mat orig;
	for(int i = 0; i < SAMPLE_SIZE; i++)
	{
		cv::Mat img(IMG_HEIGHT, IMG_WIDTH, CV_8UC1, img_data + IMG_HEIGHT * IMG_WIDTH * i);
		sprintf_s(file_path, "%soriginal\\%d.jpg", file_path_root, i + 1);
		orig = cv::imread(file_path);
		cv::cvtColor(orig, img, cv::COLOR_BGR2GRAY);
		samples.push_back(img);
	}

	char textbuf[512];
	cv::Point2d pointbuf[4];
	FILE* fp;
	sprintf_s(file_path, "%slabel.csv", file_path_root);
	fopen_s(&fp, file_path, "rt");
	for(int i = 0; i < SAMPLE_SIZE; i++)
	{
		std::fgets(textbuf, sizeof(textbuf) - 1, fp);
		sscanf_s(textbuf, "%d|%d,%d|%d,%d|%d,%d|%d",
				 &(pointbuf[0].x), &(pointbuf[0].y), &(pointbuf[1].x), &(pointbuf[1].y),
				 &(pointbuf[2].x), &(pointbuf[2].y), &(pointbuf[3].x), &(pointbuf[3].y));

		labels.push_back(pointbuf[0]);
		labels.push_back(pointbuf[1]);
		labels.push_back(pointbuf[2]);
		labels.push_back(pointbuf[3]);
	}
	fclose(fp);
}


void SampleManager::showSamples()
{
	for(int i = 0; i < SAMPLE_SIZE; i++)
	{
		cv::imshow("Showing samples", samples[i]);
		if(cv::waitKey() == 27) break;
	}
}


void SampleManager::detectEdges()
{
	CudaModule* cm = CudaModule::getInstance();

	cm->upLoadImg(img_data);
	cm->hvSobelFilter();
	cm->downLoadEdge(img_data);
}


void SampleManager::saveEdges()
{
	char file_path[512];
	for(int i = 0; i < SAMPLE_SIZE; i++)
	{
		sprintf_s(file_path, "%sedge\\%d.bmp", file_path_root, i + 1);
		cv::imwrite(file_path, samples[i]);
	}
}


void SampleManager::loadEdges()
{
	char file_path[512];
	cv::Mat orig;
	for(int i = 0; i < SAMPLE_SIZE; i++)
	{
		cv::Mat img(IMG_HEIGHT, IMG_WIDTH, CV_8UC1, img_data + IMG_HEIGHT * IMG_WIDTH * i);
		sprintf_s(file_path, "%sedge\\%d.bmp", file_path_root, i + 1);
		orig = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
		orig.copyTo(img);
		samples.push_back(img);
	}

	char textbuf[512];
	cv::Point2i pointbuf[4];
	FILE* fp;
	sprintf_s(file_path, "%slabel.csv", file_path_root);
	fopen_s(&fp, file_path, "rt");
	for(int i = 0; i < SAMPLE_SIZE; i++)
	{
		std::fgets(textbuf, sizeof(textbuf) - 1, fp);
		sscanf_s(textbuf, "%d|%d,%d|%d,%d|%d,%d|%d",
				 &(pointbuf[0].x), &(pointbuf[0].y), &(pointbuf[1].x), &(pointbuf[1].y),
				 &(pointbuf[2].x), &(pointbuf[2].y), &(pointbuf[3].x), &(pointbuf[3].y));

		labels.push_back(pointbuf[0]);
		labels.push_back(pointbuf[1]);
		labels.push_back(pointbuf[2]);
		labels.push_back(pointbuf[3]);
	}
	fclose(fp);
}


void SampleManager::circleHoughTransform()
{
	CudaModule* cm = CudaModule::getInstance();

	cm->upLoadEdge(img_data);
	cm->circleHoughTransform();
	//cm->downLoadBinArray(binarray_data);

	/*
	for(int i = 0; i < SAMPLE_SIZE; i++)
	{
		int count = 0;
		for(int j = 0; j < IMG_HEIGHT * IMG_WIDTH; j++)
		{
			//if(img_data[IMG_HEIGHT * IMG_WIDTH * i + j] >= 30) count++;
			if(img_data[IMG_HEIGHT * IMG_WIDTH * i + j] < 30)
				img_data[IMG_HEIGHT * IMG_WIDTH * i + j] = 0;
			else img_data[IMG_HEIGHT * IMG_WIDTH * i + j] = 255;
		}
		printf("Sample %d : %d\n", i, count);
	}
	
	memset(img_data, 0, IMG_HEIGHT * IMG_WIDTH * SAMPLE_SIZE);
	for(int i = 0; i < SAMPLE_SIZE; i++)
	{
		int cardinality = binarray_data[i * 16384];
		printf("Sample %d : %d\n", i, cardinality);
		if(i == 99) return;//printf("card - %d : ", cardinality);
		int count = 0;
		for(int j = 0; j < cardinality; j++)
		{
			int thread_num = binarray_data[i * 16384 + j];
			int x = thread_num % IMG_WIDTH;
			int y = thread_num / IMG_WIDTH;
			if(i == 99)
			{
				printf("%d (%d %d), ", j, x, y);
			}
			img_data[IMG_HEIGHT * IMG_WIDTH * i + x + y * IMG_WIDTH] = 255;
		}
	}
	*/
	cm->reduceDimension();
	cm->downLoadScore2d(img_data);
}


void SampleManager::testResults()
{
	int count = 0;
	for(int i = 0; i < SAMPLE_SIZE; i++)
	{
		printf("Testing %d...", i + 1);
		Detector det;
		for(int j = 0; j < IMG_WIDTH * IMG_HEIGHT; j++)
		{
			int score = img_data[j + i * IMG_WIDTH * IMG_HEIGHT];
			det.accumulate(j % IMG_WIDTH, j / IMG_WIDTH, score);
		}
		int cur_correct = det.test(labels[i * 4].x, labels[i * 4].y, labels[i * 4 + 1].x, labels[i * 4 + 1].y,
								   labels[i * 4 + 2].x, labels[i * 4 + 2].y, labels[i * 4 + 3].x, labels[i * 4 + 3].y);
		printf("%d correct!\n", cur_correct);
		count += cur_correct;
	}

	printf("Total %d corrected!\n", count);
}

