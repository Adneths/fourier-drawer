#pragma once

#include "core.h"
#include "FourierSeries.h"
#include "LineStrip.h"
#include "CyclicalLineStrip.h"
#include "NumCpp.hpp"


class NpForuierSeries : public FourierSeries {
public:
	NpForuierSeries(LineStrip* vectorLine, CyclicalLineStrip* pathLine, std::complex<float>* series, size_t size, float dt, size_t cacheSize);
	~NpForuierSeries();
	float increment(size_t count);
	void updateBuffers();
private:
	float dt;
	nc::NdArray<std::complex<float>> vector, step, pathCache;
	size_t cacheSize;
	LineStrip* vectorLine;
	CyclicalLineStrip* pathLine;
};