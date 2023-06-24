#include "core.h"

struct RenderParam {
	float x, y, width, height;
	float vectorWidth, trailWidth;
	glm::vec3 vectorColor, trailColor;
	int fps;
	const char* output;
	bool followPath, trailFade;
};