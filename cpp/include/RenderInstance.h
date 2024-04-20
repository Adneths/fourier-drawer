#pragma once

#include "RenderParam.h"
#include "MultiBuffer.h"
#include "VideoEncoder.h"
#include "Lines.h"
#include "LineStrip.h"
#include "Rect.h"
#include "core.h"
#include <vector>

class RenderInstance {
private:
	const char* output_name;
	int width, height;

	VideoEncoder* encoder;
	uint8_t* frameraw; size_t framebytes;
	MultiBuffer* multiBuffer;

	GLuint solidShader, fadeShader;

	LineStrip* vector;
	Lines* trail;

	struct ViewInstance {
		bool valid;
		int id;
		glm::mat3 viewMtx, invViewMtx;
		float offsetX, offsetY, xScale, yScale, zoom;
		glm::vec3 vectorColor, pathColor, backgroundColor, borderColor;
		float vectorWidth, pathWidth, borderWidth;

		bool followPath, pathFade, drawBackground, drawBorderOnOther;

		Rect* rect;
	} views[8];

	std::vector<ViewInstance*> globalBorders;
public:

	RenderInstance(RenderParam params, GLuint solidShader, GLuint fadeShader, LineStrip* vector, Lines* trail, float width, float height);
	~RenderInstance();
	void draw(const float& time, glm::vec2* pos);
	void postDraw();
	void encode();
};