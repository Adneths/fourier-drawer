#pragma once

#include "core.h"

using namespace math;

template <typename T>
class FourierSeries {
public:
	bool invalid = false;
	virtual T increment(size_t count, T time) = 0;
	virtual void updateBuffers(vec2<T>* vecHeadPtr = nullptr) = 0;
	virtual void readyBuffers() = 0;
	virtual void resetTrail(vec2<T>* vecHeadPtr = nullptr) = 0;
	virtual void init(T time) = 0;
	bool valid()
	{
		return !invalid;
	}
};