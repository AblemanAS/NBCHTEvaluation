
#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include "commons.hpp"

typedef struct
{
	std::vector<cv::Point3i> points;
	cv::Point2i avg_position;
	cv::Point2i total_position;
	int total_score = 0;
} Cluster;

class Detector
{
public:
	void accumulate(int x, int y, int score);
	int test(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4);

private:
	std::vector<cv::Point3i> points;
	std::vector<Cluster> clusters;
};

#endif DETECTOR_HPP