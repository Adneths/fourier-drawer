#pragma once

#include "core.h"
#include "FourierSeries.h"
#include "LineStrip.h"
#include "Lines.h"
#include "NumCpp.hpp"


class NpForuierSeries : public FourierSeries {
public:
	NpForuierSeries(LineStrip* vectorLine, Lines* pathLine, std::complex<float>* mags, int* freqs, size_t size, float dt, size_t cacheSize);
	~NpForuierSeries();
	float increment(size_t count, float time) override;
	void updateBuffers() override;
	void readyBuffers(glm::vec2* vecHeadPtr = nullptr) override;
	void resetTrail() override;
	void init(float time) override;
private:
	size_t head;
	float last[3];

	nc::NdArray<std::complex<float>> vector, step;
	nc::NdArray<int> freqsArr;
	float dt;
	float* pathCache;
	size_t cacheSize;
	LineStrip* vectorLine;
	Lines* pathLine;

	size_t lineWidth, cacheFloatSize, pathBufferSize;
};