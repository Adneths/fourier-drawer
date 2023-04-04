#pragma once

#include "core.h"

class LineStrip {
protected:
	GLuint VBO, VAO;
	glm::vec3 color;
	size_t count;
public:
	LineStrip(glm::vec2 vertex, size_t count, glm::vec3 color);
	LineStrip(float* vertices, size_t length, glm::vec3 color);
	virtual void draw(GLuint shader, glm::mat3 viewMtx);
	virtual ~LineStrip();

	virtual GLuint getBuffer();
	virtual void finish();
	size_t getCount();
};