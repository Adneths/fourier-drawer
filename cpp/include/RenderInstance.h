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
	RenderParam params;

	GLuint vectorShader, pathShader;
	glm::mat3 viewMtx;

	LineStrip* vector;
	Lines* trail;
public:
	RenderInstance(RenderParam params, GLuint vectorShader, GLuint pathShader, LineStrip* vector, Lines* trail);
	~RenderInstance();
	GLsync draw(float time);
	void encode();
};