#pragma once


class FourierSeries {
public:
	virtual float increment(size_t count, float time) = 0;
	virtual void updateBuffers() = 0;
	virtual void readyBuffers() = 0;
	virtual void resetTrail() = 0;
	virtual void init(float time) = 0;
};