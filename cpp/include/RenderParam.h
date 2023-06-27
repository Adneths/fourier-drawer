#include "core.h"
#include <iostream>

struct RenderParam {
	float x, y, width, height, zoom;
	float vectorWidth, trailWidth;
	glm::vec3 vectorColor, trailColor;
	int fps;
	const char* output;
	bool followPath, trailFade;
};

inline std::ostream& operator<<(std::ostream& os, const RenderParam& p) {
	return os << "{ x=" << p.x << ", y=" << p.y << ", width=" << p.width << ", height="
		<< p.height << ", zoom=" << p.zoom << ", vectorWidth=" << p.vectorWidth << ", trailWidth=" << p.trailWidth
		<< ", vectorColor=(" << p.vectorColor.x << ", " << p.vectorColor.y << ", " << p.vectorColor.z
		<< "), trailColor=(" << p.trailColor.x << ", " << p.trailColor.y << ", " << p.trailColor.z
		<< "), fps=" << p.fps << ", output=" << p.output << ", followPath=" << p.followPath
		<< ", trailFade=" << p.trailFade << " }";
}