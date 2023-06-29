#pragma once

#include <glm/glm.hpp>

class FourierSeries {
public:
	virtual float increment(size_t count, float time) = 0;
	virtual void updateBuffers() = 0;
	virtual void readyBuffers(glm::vec2* vecHeadPtr = nullptr) = 0;
	virtual void resetTrail() = 0;
	virtual void init(float time) = 0;
};