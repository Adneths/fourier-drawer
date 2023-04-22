#pragma once


class FourierSeries {
public:
	virtual float increment(size_t count, float time) = 0;
	virtual void updateBuffers() = 0;
};