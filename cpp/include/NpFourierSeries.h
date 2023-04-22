#pragma once

#include "core.h"
#include "FourierSeries.h"
#include "LineStrip.h"
#include "Lines.h"
#include "NumCpp.hpp"


class NpForuierSeries : public FourierSeries {
public:
	NpForuierSeries(LineStrip* vectorLine, Lines* pathLine, std::complex<float>* series, size_t size, float dt, size_t cacheSize);
	~NpForuierSeries();
	float increment(size_t count, float time);
	void updateBuffers();
private:
	size_t head;
	float last[3];

	float dt;
	nc::NdArray<std::complex<float>> vector, step;
	float* pathCache;
	size_t cacheSize;
	LineStrip* vectorLine;
	Lines* pathLine;

	size_t lineWidth, cacheFloatSize, pathBufferSize;
};