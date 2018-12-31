
#include "commons.hpp"
#include "SampleManager.hpp"

#define FILE_PATH "D:\\ExperimentData\\"

int main()
{
	srand(0);
	SampleManager sm = SampleManager(FILE_PATH);

	//sm.generateSamples();

	//sm.loadSamples();
	//sm.detectEdges();
	//sm.saveEdges();
	
	sm.loadEdges();
	sm.circleHoughTransform();
	//sm.testResults();

	//sm.showSamples();

	return 0;
}
