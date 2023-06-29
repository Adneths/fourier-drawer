#pragma once

#include "RenderParam.h"
#include "MultiBuffer.h"
#include "VideoEncoder.h"
#include "Lines.h"
#include "LineStrip.h"
#include "core.h"

class RenderInstance {
private:
	VideoEncoder* encoder;
	uint8_t* frameraw; size_t framebytes;
	MultiBuffer* multiBuffer;

	GLuint vectorShader, pathShader;
	glm::mat3 viewMtx;
	float offsetX, offsetY;

	LineStrip* vector;
	Lines* trail;
public:
	const RenderParam params;

	RenderInstance(RenderParam params, GLuint vectorShader, GLuint pathShader, LineStrip* vector, Lines* trail, float width, float height);
	~RenderInstance();
	GLsync draw(const float& time, glm::vec2* pos);
	void encode();
};