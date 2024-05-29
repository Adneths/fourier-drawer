#pragma once

#include "RenderParam.h"
#include "MultiBuffer.h"
#include "VideoEncoder.h"
#include "Lines.h"
#include "LineStrip.h"
#include "Rect.h"
#include "core.h"
#include <vector>

using namespace math;

template <typename T>
struct ViewInstance {
	bool valid;
	int id;
	mat3<T> viewMtx, invViewMtx;
	T offsetX, offsetY, xScale, yScale, zoom;
	fvec3 vectorColor, pathColor, backgroundColor, borderColor;
	float vectorWidth, pathWidth, borderWidth;

	bool followPath, pathFade, drawBackground, drawBorderOnOther;

	Rect<T>* rect;
};

template <typename T>
class RenderInstance {
private:
	const char* output_name;
	int width, height;

	VideoEncoder* encoder;
	uint8_t* frameraw; size_t framebytes;
	MultiBuffer* multiBuffer;

	GLuint solidShader, fadeShader;

	LineStrip<T>* vector;
	Lines<T>* trail;

	ViewInstance<T> views[8];

	std::vector<ViewInstance<T>*> globalBorders;
public:
	RenderInstance(RenderParam params, GLuint solidShader, GLuint fadeShader, LineStrip<T>* vector, Lines<T>* trail, int width, int height);
	~RenderInstance();
	void draw(const T& time, vec2<T>* pos);
	void postDraw();
	void encode();
};