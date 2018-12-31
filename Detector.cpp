
#include "Detector.hpp"

#define CLUSTER_TOLERANCE	10


void Detector::accumulate(int x, int y, int score)
{
	points.push_back(cv::Point3i(x, y, score));
}


int Detector::test(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4)
{
	std::sort(points.begin(), points.end(), [](cv::Point3i a, cv::Point3i b) { return a.z > b.z; });
	
	for(int e = 0; e < 500; e++)
	{
		int x = points[e].x;
		int y = points[e].y;
		int score = points[e].z;
		int found = 0;
		if(score < 1) break;
		for(int i = 0; i < clusters.size(); i++)
		{
			int distx = clusters[i].avg_position.x - x;
			int disty = clusters[i].avg_position.y - y;
			if(distx * distx + disty * disty < CLUSTER_TOLERANCE * CLUSTER_TOLERANCE)
			{
				clusters[i].points.push_back(cv::Point3i(x, y, score));
				clusters[i].total_score += score;
				clusters[i].total_position.x += x * score;
				clusters[i].total_position.y += y * score;
				clusters[i].avg_position.x = clusters[i].total_position.x / clusters[i].total_score;
				clusters[i].avg_position.y = clusters[i].total_position.y / clusters[i].total_score;
				found = 1;
				break;
			}
		}

		// Not found
		if(!found)
		{
			Cluster cur_cluster;
			cur_cluster.points.push_back(cv::Point3i(x, y, score));
			cur_cluster.total_score = score;
			cur_cluster.total_position.x = x * score;
			cur_cluster.total_position.y = y * score;
			cur_cluster.avg_position.x = x;
			cur_cluster.avg_position.y = y;
			clusters.push_back(cur_cluster);
		}
	}


	std::sort(clusters.begin(), clusters.end(), [](Cluster a, Cluster b) { return a.total_score > b.total_score; });
	printf("\ncluster : %d\n", clusters.size());
	int correct_count = 0;
	float distx, disty;
	int search_size = clusters.size() < 4 ? clusters.size() : 4;
	for(int i = 0; i < search_size; i++)
	{
		printf("(%d, %d) : %d, ", clusters[i].avg_position.x, clusters[i].avg_position.y, clusters[i].total_score);
		distx = clusters[i].avg_position.x - x1;
		disty = clusters[i].avg_position.y - y1;
		if(distx * distx + disty * disty < CLUSTER_TOLERANCE * CLUSTER_TOLERANCE) correct_count++;
		distx = clusters[i].avg_position.x - x2;
		disty = clusters[i].avg_position.y - y2;
		if(distx * distx + disty * disty < CLUSTER_TOLERANCE * CLUSTER_TOLERANCE) correct_count++;
		distx = clusters[i].avg_position.x - x3;
		disty = clusters[i].avg_position.y - y3;
		if(distx * distx + disty * disty < CLUSTER_TOLERANCE * CLUSTER_TOLERANCE) correct_count++;
		distx = clusters[i].avg_position.x - x4;
		disty = clusters[i].avg_position.y - y4;
		if(distx * distx + disty * disty < CLUSTER_TOLERANCE * CLUSTER_TOLERANCE) correct_count++;
	}
	printf("\n");

	return correct_count;
}

