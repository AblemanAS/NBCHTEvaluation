#ifndef SAMPLE_MANAGER_HPP
#define SAMPLE_MANAGER_HPP

#include "commons.hpp"

class SampleManager
{
public:
	SampleManager(char* file_path);
	void generateSamples();
	void loadSamples();
	void showSamples();
	void detectEdges();
	void saveEdges();
	void loadEdges();
	void circleHoughTransform();
	void testResults();

private:
	char* file_path_root;
	uchar* img_data;
	int* binarray_data;
	std::vector<cv::Mat> samples;
	std::vector<cv::Point2i> labels;
};

#endif SAMPLE_MANAGER_HPP