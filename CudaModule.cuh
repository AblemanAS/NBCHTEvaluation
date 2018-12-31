
#ifndef CUDA_MODULE
#define CUDA_MODULE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "commons.hpp"

class CudaModule
{
public:
	inline static CudaModule* getInstance() { if(!self) { self = new CudaModule; } return self; }
	void initialize();
	void hvSobelFilter();
	void circleHoughTransform();
	void reduceDimension();
	void sync();

	void upLoadImg(uchar* data);
	void upLoadEdge(uchar* data);
	void downLoadEdge(uchar* data);
	void downLoadScore2d(uchar* data);
	void downLoadBinArray(int* data);

private:
	CudaModule();
	static CudaModule* self;
	unsigned char* imgdata_dev;
	unsigned char* imgdata_padded_dev;		// The full space (8 pixel padding for filtering)
	unsigned char* edgedata_dev;
	unsigned char* edgedata_padded_dev;
	unsigned char* score3ddata_dev;
	unsigned char* score2ddata_dev;
	int* binarray_dev;
};

#endif // !CUDA_MODULE
