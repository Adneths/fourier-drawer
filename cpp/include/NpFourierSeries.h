#pragma once

#include "core.h"
#include "FourierSeries.h"
#include "LineStrip.h"
#include "Lines.h"
#include "NumCpp.hpp"

using namespace math;

template <typename T>
class NpFourierSeries : public FourierSeries<T> {
public:
	NpFourierSeries(LineStrip<T>* vectorLine, Lines<T>* pathLine, std::complex<T>* mags, int* freqs, size_t size, T dt, size_t cacheSize);
	~NpFourierSeries();
	T increment(size_t count, T time) override;
	void updateBuffers(vec2<T>* vecHeadPtr = nullptr) override;
	void readyBuffers() override;
	void resetTrail(vec2<T>* vecHeadPtr = nullptr) override;
	void init(T time) override;
private:
	size_t head;
	T last[3];

	nc::NdArray<std::complex<T>> vector, step;
	nc::NdArray<int> freqsArr;
	T dt;
	T* pathCache;
	size_t cacheSize;
	LineStrip<T>* vectorLine;
	Lines<T>* pathLine;

	size_t lineWidth, cacheFloatSize, pathBufferSize;
};